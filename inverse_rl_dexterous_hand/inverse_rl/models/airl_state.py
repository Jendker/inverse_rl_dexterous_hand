# Based on https://github.com/justinjfu/inverse_rl

import math
import os
import tensorflow as tf
from inverse_rl_dexterous_hand.inverse_rl.adversarial_samples import AdversarialSamples
from inverse_rl_dexterous_hand.inverse_rl.noise_samples_generator import NoiseSamplesGenerator
from tensorflow.keras import Model
import numpy as np
from gym.spaces.box import Box
from gym.spaces.utils import flatdim
import copy
from inverse_rl_dexterous_hand.inverse_rl.models.fusion_manager import RamFusionDistr, DiskFusionDistr
from inverse_rl_dexterous_hand.inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl_dexterous_hand.inverse_rl.models.architectures import relu_net, normalized_relu_net
from inverse_rl_dexterous_hand.inverse_rl.utils import TrainingIterator


class Discriminator(Model):
    def __init__(self, env,
                 reward_arch=normalized_relu_net,
                 reward_arch_args=None,
                 value_fn_arch=relu_net,
                 discount=1.0,
                 state_only=False,
                 normalization_learning_rate=0,
                 visible_indices=None,
                 feature_selection=None,
                 use_timestamp=False,
                 alg_use_timestamp=False,
                 initialisation_value=0):
        super(Discriminator, self).__init__()
        if reward_arch_args is None:
            reward_arch_args = {}

        assert not(use_timestamp and alg_use_timestamp), "Either specify IRL use_timestamp or train use_timestamp"
        self.dO = flatdim(env.observation_space)
        self.dU = flatdim(env.action_space)
        assert isinstance(env.action_space, Box)
        self.gamma = discount
        assert value_fn_arch is not None
        self.state_only = state_only

        rew_input_size = self.dO
        if not self.state_only:
            rew_input_size = self.dO + self.dU
        value_fn_input_size = self.dO
        if use_timestamp or alg_use_timestamp:
            rew_input_size += 1
            value_fn_input_size += 1
        self.reward = reward_arch(din=rew_input_size, dout=1, initialisation_value=initialisation_value,
                                  **reward_arch_args)
        self.fitted_value_fn = value_fn_arch(din=value_fn_input_size, dout=1, d_hidden=48)
        if isinstance(visible_indices, str) and visible_indices:
            evaluated = eval(visible_indices)
            visible_indices = list(range(self.dO)[evaluated])
            if alg_use_timestamp and isinstance(evaluated, slice):
                if evaluated.stop is None:  # means goes to the end - we need to shift it by one, because end will be the appended index
                    for index, value in enumerate(visible_indices):
                        visible_indices[index] = value - 1

        if (use_timestamp or alg_use_timestamp) and visible_indices:
            to_append = self.dO
            if alg_use_timestamp:
                to_append -= 1
            visible_indices.append(to_append) # d0 will be the first index after observation space - timestamp, needs to be added
        self.visible_indices = visible_indices
        self.feature_selection = feature_selection
        self.normalization_learning_rate = normalization_learning_rate

    def call(self, x, **kwargs):
        act_t, obs_t, nobs_t, lprobs = x
        if 'rew_input' in kwargs:
            rew_input = kwargs['rew_input']
        else:
            rew_input = self.rew_input(obs_t, act_t)
        reward = self.reward(rew_input)
        log_p_tau = reward + self.gamma * self.fitted_value_fn(nobs_t) - self.fitted_value_fn(obs_t)

        log_q_tau = lprobs
        log_pq = tf.math.reduce_logsumexp(tf.stack([log_p_tau, log_q_tau]), axis=0)
        discrim_output = tf.math.exp(log_p_tau - log_pq)
        return discrim_output, log_p_tau, log_pq, log_q_tau

    def update_normalization_from_rewards(self, act_t, obs_t, expert_act_t, expert_obs_t):
        rewards = self._eval_reward_unnormalized(obs_t=np.concatenate([obs_t, expert_obs_t]),
                                                 act_t=np.concatenate([act_t, expert_act_t]))
        self.reward.mean = (1 - self.normalization_learning_rate) * self.reward.mean +\
                            self.normalization_learning_rate * np.mean(rewards)
        self.reward.std_dev = (1 - self.normalization_learning_rate) * self.reward.std_dev +\
                               self.normalization_learning_rate * np.std(rewards)
        rewards = (rewards - self.reward.mean.numpy()) / self.reward.std_dev.numpy()
        return rewards

    def calculate_loss(self, log_p_tau, log_pq, log_q_tau, labels):
        loss = -tf.reduce_mean(labels * (log_p_tau - log_pq) + (1 - labels) * (log_q_tau - log_pq))
        return loss

    def get_parameters(self):
        from itertools import chain
        return chain(self.reward.parameters(), self.fitted_value_fn.parameters())

    def rew_input(self, obs_t, act_t):
        if self.visible_indices:
            if isinstance(obs_t, tf.Tensor):
                obs_t = obs_t.numpy()
            trimmed_obs = obs_t[:, self.visible_indices]
            rew_input = trimmed_obs
            if not self.state_only:
                rew_input = np.concatenate([trimmed_obs, act_t], axis=1)
        else:
            rew_input = obs_t
            if not self.state_only:
                rew_input = np.concatenate([obs_t, act_t], axis=1)
        if self.feature_selection is not None:
            rew_input = self.feature_selection.transform(rew_input)
        return tf.convert_to_tensor(rew_input)

    def _eval_reward(self, obs_t, act_t):
        reward = self.reward(self.rew_input(obs_t, act_t))
        return reward

    def _eval_reward_unnormalized(self, obs_t, act_t):
        reward = self.reward(self.rew_input(obs_t, act_t), normalize=False)
        return reward

    def get_value_f(self, obs_t):
        return self.fitted_value_fn.predict(obs_t)

    def logger_outputs(self, act_t, obs_t, nobs_t, lprobs):
        reward = self.reward(self.rew_input(obs_t, act_t))
        fitted_value = self.fitted_value_fn(obs_t)
        log_p_tau = reward + self.gamma * self.fitted_value_fn(nobs_t) - fitted_value

        log_q_tau = lprobs
        log_pq = tf.math.reduce_logsumexp(tf.stack([log_p_tau, log_q_tau]), axis=0)
        discrim_output = tf.math.exp(log_p_tau - log_pq)
        return reward, fitted_value, discrim_output


class GeneralizingCheckpoint:
    def __init__(self):
        self.iteration_to_load = 0
        self.iteration_save_freq = None
        self.first_save_iteration = None
        self.min_loss_considered_still_generalizing = 0.2
        self.min_fitting_steps_to_not_load = 4
        self.last_loaded_generalizing_checkpoint = None
        self.last_loss = 0
        self.iteration_path = None

    def check_load_generalizing_checkpoint(self, fitting_iteration_number, outer_loop_iteration_number, airl_model):
        if fitting_iteration_number < self.min_fitting_steps_to_not_load:
            if self.last_loaded_generalizing_checkpoint is not None and self.last_loaded_generalizing_checkpoint >= self.iteration_to_load:
                if self.iteration_save_freq is None:
                    print("GeneralizingCheckpoint: Iteration save frequency not known yet. Could not backstep the "
                          "generalizing checkpoint number")
                else:
                    if self.iteration_to_load - self.iteration_save_freq > 0:
                        self.iteration_to_load -= self.iteration_save_freq
                        print("GeneralizingCheckpoint: Backstep the generalizing checkpoint number to", str(self.iteration_to_load))
            if self.iteration_path is not None and self.iteration_to_load > 0:
                airl_model.load_iteration(self.iteration_path, self.iteration_to_load)
                print('GeneralizingCheckpoint: Reward function does not generalize. Reloading previous generalizing '
                      'checkpoint with iteration number', self.iteration_to_load)
                self.last_loaded_generalizing_checkpoint = outer_loop_iteration_number

    def new_checkpoint(self, iteration, iteration_path):
        if self.iteration_save_freq is None:
            if self.first_save_iteration is None:
                self.first_save_iteration = iteration
            else:
                self.iteration_save_freq = iteration - self.first_save_iteration
        if self.last_loss >= self.min_loss_considered_still_generalizing:
            self.iteration_to_load = iteration
        self.last_loss = 0
        if self.iteration_path is None:
            self.iteration_path = os.path.dirname(os.path.dirname(iteration_path))


class AIRL(SingleTimestepIRL):
    """ 
    Args:
        fusion: Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """
    def __init__(self, env,
                 expert_trajs=None,
                 reward_arch=normalized_relu_net,
                 reward_arch_args=None,
                 value_fn_arch=relu_net,
                 score_discrim=False,
                 discount=1.0,
                 state_only=False,
                 max_itrs=100,
                 fusion=None,
                 initialization_job=None,
                 normalization_learning_rate=0,
                 augment_samples_times=0,
                 augment_expert_times=0,
                 lower_sample_augmentation_count=True,
                 visible_indices=None,
                 PCA_components=None,
                 PCA_scaling=False,
                 mitra_k=None,
                 transformation_matrix=None,
                 seed=None,
                 dump_paths_percentage=None,
                 adversarial_samples_percentage=0,
                 adversarial_samples_range=0,
                 adversarial_samples_alpha_ratio=0,
                 use_timestamp=False,
                 alg_use_timestamp=False,
                 discr_initialisation_value=0,
                 noise_samples_generator_args=None):
        super(AIRL, self).__init__()
        if reward_arch_args is None:
            reward_arch_args = {}

        if seed is not None:
            tf.random.set_seed(seed)

        assert fusion is None or dump_paths_percentage is None, "Set either fusion or dump_paths_percentage"
        self.dump_paths_percentage = dump_paths_percentage
        if fusion is not None:
            if fusion == 'disk':
                self.fusion = DiskFusionDistr(subsample_ratio=0.2)
            elif fusion == 'ram':
                self.fusion = RamFusionDistr(buf_size=10000, subsample_ratio=0.5)
            else:
                raise ValueError("Fusion parameter has to be either 'disk' or 'ram'. Given:", fusion)
        elif self.dump_paths_percentage is not None:
            self.fusion = DiskFusionDistr(subsample_ratio=0.5)
        else:
            self.fusion = None
        self.score_discrim = score_discrim
        assert value_fn_arch is not None
        self.max_itrs = max_itrs
        feature_selection = None
        assert(not(visible_indices is not None and PCA_components is not None)), 'Must not specify visible_indices '\
            'and PCA_components at the same time.'
        if PCA_components is not None:
            pca = FPPCA(n_components=PCA_components, scaling=PCA_scaling)
            reward_input = []
            if state_only:
                for demo in expert_trajs:
                    reward_input.extend(demo['observations'])
            else:
                # TODO: Improve PCA for observations and actions, should probably do PCA for both separaterly
                for demo in expert_trajs:
                    reward_input.extend(np.concatenate([demo['observations'], demo['actions']], axis=1))
            reward_input = np.array(reward_input)
            pca.fit(reward_input)
            feature_selection = pca
        elif mitra_k is not None:
            mitra = Mitra(k=mitra_k)
            reward_input = []
            if state_only:
                for demo in expert_trajs:
                    reward_input.extend(demo['observations'])
            else:
                # TODO: Improve PCA for observations and actions, should probably do PCA for both separaterly
                for demo in expert_trajs:
                    reward_input.extend(np.concatenate([demo['observations'], demo['actions']], axis=1))
            reward_input = np.array(reward_input)
            mitra.fit(reward_input)
            feature_selection = mitra
        elif transformation_matrix is not None:
            pca = FPPCA(transformation_matrix=transformation_matrix)
            feature_selection = pca
        self.discriminator = Discriminator(env, reward_arch, reward_arch_args, value_fn_arch, discount, state_only,
                                           normalization_learning_rate, visible_indices,
                                           feature_selection=feature_selection, use_timestamp=use_timestamp,
                                           initialisation_value=discr_initialisation_value, alg_use_timestamp=alg_use_timestamp)
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, net=self.discriminator,
                                              normalization_mean=self.discriminator.reward.mean,
                                              normalization_std_dev=self.discriminator.reward.std_dev)
        self.manager = None
        if initialization_job:
            print("Trying initialization from job", initialization_job, "...")
            initialization_path = os.path.join('Runs', initialization_job, 'iterations/discriminator')
            abs_initialization_path = os.path.abspath(initialization_path)
            self.checkpoint.restore(tf.train.latest_checkpoint(abs_initialization_path))
            print("Initialization successful.")
        self.augment_samples_times = augment_samples_times
        if self.augment_samples_times > 0:
            from inverse_rl_dexterous_hand.inverse_rl.augmentation import Augmentation
            self.augmentation = Augmentation(env)
            self.lower_augmentation_count = lower_sample_augmentation_count
            if augment_expert_times > 0:
                expert_trajs = self.augmentation.augment_paths(expert_trajs, num_cpu=4, augment_times=augment_expert_times,
                                                               angle_min_max=30)
        else:
            if augment_expert_times > 0:
                augment_samples_times = Augmentation(env)
                expert_trajs = augment_samples_times.augment_paths(expert_trajs, num_cpu=4, augment_times=augment_expert_times,
                                                                   angle_min_max=30)
            self.augmentation = None
        self.set_demos(expert_trajs)
        self.normalization = normalization_learning_rate > 1e-8
        if noise_samples_generator_args is not None and noise_samples_generator_args['samples_percent'] > 0:
            assert state_only, \
                "Only samples state reward is supported for adding noise samples generator"
            self.noise_samples_generator = NoiseSamplesGenerator(**noise_samples_generator_args, env=env)
        else:
            self.noise_samples_generator = None
        if adversarial_samples_percentage > 0:
            self.adversarial_samples_generator = AdversarialSamples(adversarial_samples_percentage, custom_range=adversarial_samples_range,
                                                                    alpha_ratio=adversarial_samples_alpha_ratio)
        else:
            self.adversarial_samples_generator = None
        self.use_timestamp = use_timestamp
        self.generalizing_checkpoint = GeneralizingCheckpoint()

    def input_size(self):
        return self.discriminator.dO

    def add_paths_to_buffer(self, paths, main_loop_step):
        if isinstance(self.fusion, RamFusionDistr):
            self.fusion.add_paths(paths)
        elif isinstance(self.fusion, DiskFusionDistr):
            self.fusion.save_itr_paths(main_loop_step, paths)

    def fit(self, paths, policy=None, batch_size=32, logger=None, lr=1e-3, minimal_loss=0.01, num_cpu='max',
            policy_updates_count=1, main_loop_step=None, main_loop_percentage=None, **kwargs):
        self.optimizer.lr.assign(lr)
        if self.augmentation is not None:
            if self.lower_augmentation_count:
                augment_times = int(np.ceil(self.augment_samples_times) / policy_updates_count)
            else:
                augment_times = self.augment_samples_times
            paths = self.augmentation.augment_paths(paths, num_cpu=num_cpu, augment_times=augment_times)
        if self.fusion is not None and self.dump_paths_percentage is None:
            paths_len = len(paths)
            target_len = int(self.max_itrs * batch_size / 2)
            if logger:
                logger.log_kv('IRLBufferSize', self.fusion.buffer_size())
            old_paths = self.fusion.sample_paths(n=target_len)
            self.add_paths_to_buffer(paths, main_loop_step)
            # make 50:50 old and new paths
            paths = (paths * max(math.ceil(target_len / len(paths)), 1))[:target_len] + old_paths
            if len(paths) < self.max_itrs * batch_size:
                paths += paths
            paths = paths[:self.max_itrs * batch_size]
        elif self.dump_paths_percentage is not None:
            if main_loop_percentage > self.dump_paths_percentage:
                self.add_paths_to_buffer(paths, main_loop_step)
        if self.noise_samples_generator is not None and paths:
            noise_samples = self.noise_samples_generator.get_noise_samples(paths)
            paths.extend(noise_samples)

        # eval samples under current policy
        self._compute_path_probs(paths, insert=True)

        if self.use_timestamp:
            expert_trajs = copy.deepcopy(self.expert_trajs)
        else:
            expert_trajs = self.expert_trajs

        # eval expert log probs under current policy
        self.eval_expert_probs(expert_trajs, policy, insert=True)

        if self.use_timestamp:
            self.insert_timestamps(paths)
            self.insert_timestamps(expert_trajs)

        self._insert_next_state(paths)
        self._insert_next_state(expert_trajs)

        obs, obs_next, acts, acts_next, path_probs = \
            self.extract_paths(paths,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs = \
            self.extract_paths(expert_trajs,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))

        path_index_batches = []
        if len(paths) >= self.max_itrs * batch_size:
            # TODO: clean this up
            indices = np.arange(len(paths))
            np.random.shuffle(indices)
            for start_idx in range(0, len(paths) - batch_size + 1, batch_size):
                path_index_batches.append(indices[start_idx:start_idx + batch_size])
            assert len(path_index_batches) == self.max_itrs and len(path_index_batches[-1]) == batch_size

        # Train discriminator
        iteration_step = 0
        for iteration_step, it in enumerate(TrainingIterator(self.max_itrs, heartbeat=5)):
            if path_index_batches:
                nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                    self.sample_batch(obs_next, obs, acts_next, acts, path_probs,
                                      path_index_batches=path_index_batches[iteration_step])
            else:
                nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                    self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=batch_size)

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs, batch_size=batch_size)

            # Build feed dict
            labels = np.zeros((batch_size*2, 1))
            labels[batch_size:] = 1.0
            if self.adversarial_samples_generator is not None:
                # get only negatives
                feed_dict = {
                    'act_t': act_batch,
                    'obs_t': obs_batch,
                    'nobs_t': nobs_batch,
                    'lprobs': lprobs_batch
                }
                # we add only negatives adversarial samples, so all labels=0
                obs_batch, act_batch, nobs_batch, _ = self.adversarial_samples_generator.add_adversarial_samples(
                    self.discriminator, feed_dict, labels=np.zeros((batch_size, 1)), use_timestamp=self.use_timestamp)
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(np.float32)
            feed_dict = {
                'act_t': act_batch,
                'obs_t': obs_batch,
                'nobs_t': nobs_batch,
                'lprobs': lprobs_batch
                }
            inputs = [act_batch, obs_batch, nobs_batch, lprobs_batch]
            with tf.GradientTape() as tape:
                discrim_output, log_p_tau, log_pq, log_q_tau = self.discriminator(inputs)
                loss = self.discriminator.calculate_loss(log_p_tau, log_pq, log_q_tau, labels)

            scalar_loss = loss.numpy()
            it.record('loss', scalar_loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tDiscriminator loss:%f' % mean_loss)
            if loss < minimal_loss:
                print("\tLoss", scalar_loss, "is lower than min loss", minimal_loss)
                print("\tBreaking discriminator training after", iteration_step, "steps")
                if 'mean_loss' not in locals():
                    mean_loss = it.pop_mean('loss')
                self.generalizing_checkpoint.check_load_generalizing_checkpoint(iteration_step, main_loop_step, self)
                break
            self.generalizing_checkpoint.last_loss = loss
            gradients = tape.gradient(loss, self.discriminator.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        if self.normalization:
            rewards = self.discriminator.update_normalization_from_rewards(act_t=acts, obs_t=obs,
                                                                           expert_act_t=expert_acts,
                                                                           expert_obs_t=expert_obs)
            logger.log_kv('normalization_mean', self.discriminator.reward.mean.numpy())
            logger.log_kv('normalization_sigma', self.discriminator.reward.std_dev.numpy())
        if logger:
            # eval value function for logging
            first_obs = [path['observations'][1, :] for path in paths]
            first_obs = np.stack(first_obs, axis=0)
            value_fs = self.discriminator.get_value_f(first_obs)
            logger.log_kv('IRLVF_mean_from_initial', np.mean(value_fs))

            logger.log_kv('IRLDiscrimLoss', mean_loss)
            #obs_next = np.r_[obs_next, np.expand_dims(obs_next[-1], axis=0)]
            rewards, logZ, dtau = self.discriminator.logger_outputs(act_t=acts, obs_t=obs, nobs_t=obs_next,
                                                                    lprobs=np.expand_dims(path_probs, axis=1))
            rewards, logZ, dtau = rewards.numpy(), logZ.numpy(), dtau.numpy()
            energy = -rewards
            logger.log_kv('IRLLogZ', np.mean(logZ))
            logger.log_kv('IRLAverageEnergy', np.mean(energy))
            logger.log_kv('IRLAverageLogPtau', np.mean(-energy-logZ))
            logger.log_kv('IRLAverageLogQtau', np.mean(path_probs))
            logger.log_kv('IRLMedianLogQtau', np.median(path_probs))
            logger.log_kv('IRLAverageDtau', np.mean(dtau))
            logger.log_kv('IRLIterations', iteration_step + 1)


            #expert_obs_next = np.r_[expert_obs_next, np.expand_dims(expert_obs_next[-1], axis=0)]
            energy, logZ, dtau = self.discriminator.logger_outputs(act_t=expert_acts, obs_t=expert_obs, nobs_t=expert_obs_next,
                                                                   lprobs=np.expand_dims(expert_probs, axis=1))
            energy, logZ, dtau = energy.numpy(), logZ.numpy(), dtau.numpy()
            energy = -energy
            logger.log_kv('IRLAverageExpertEnergy', np.mean(energy))
            logger.log_kv('IRLAverageExpertLogPtau', np.mean(-energy-logZ))
            logger.log_kv('IRLAverageExpertLogQtau', np.mean(expert_probs))
            logger.log_kv('IRLMedianExpertLogQtau', np.median(expert_probs))
            logger.log_kv('IRLAverageExpertDtau', np.mean(dtau))

        return mean_loss

    def eval(self, original_paths, **kwargs):
        """
        Return bonus
        """
        if self.use_timestamp:
            paths = self.insert_timestamps(original_paths, copy_paths=True)
        else:
            paths = original_paths
        if self.score_discrim:
            self._compute_path_probs(paths, insert=True)
            keys = ('observations', 'observations_next', 'actions', 'a_logprobs')
            for key in keys:
                if key not in paths[0]:
                    self._insert_next_state(paths)
            obs, obs_next, acts, path_probs = self.extract_paths(paths, keys=keys)
            path_probs = np.expand_dims(path_probs, axis=1)
            inputs = [acts, obs, obs_next, path_probs]
            scores, _, _, _ = self.discriminator(inputs)
            scores = scores.numpy()
            scores[scores == 1] = 1 - 1e-7
            scores[scores == 0] = 0 + 1e-7
            score = np.log(scores) - np.log(1-scores)
            score = score[:, 0]
        else:
            obs, acts = self.extract_paths(paths)
            reward = self.discriminator._eval_reward(obs, acts)
            score = reward[:, 0].numpy()
        return self.unpack(score, original_paths)

    def eval_single(self, obs, acts):
        reward = self.discriminator._eval_reward(np.expand_dims(obs, axis=0), np.expand_dims(acts, axis=0))
        score = reward.numpy().item()
        return score

    def eval_multiple(self, obs, acts):
        reward = self.discriminator._eval_reward(obs, acts)
        score = reward.numpy().squeeze()
        return score

    def save_checkpoint(self, path, iteration):
        if self.manager is None:
            full_path = os.path.join(path, 'discriminator')
            self.manager = tf.train.CheckpointManager(self.checkpoint, full_path, max_to_keep=3)
        self.discriminator.save_weights(os.path.join(path, 'discriminator_last'))
        self.discriminator.save_weights(os.path.join(path, 'discriminator', 'discriminator_weights_{}'.format(iteration)))
        self.generalizing_checkpoint.new_checkpoint(iteration, path)
        self.manager.save()

    def load_checkpoint(self, path):
        if self.manager is None:
            full_path = os.path.join(path, 'discriminator')
            self.manager = tf.train.CheckpointManager(self.checkpoint, full_path, max_to_keep=3)
        self.checkpoint.restore(self.manager.latest_checkpoint)

    def load_last(self, path):
        self.discriminator.load_weights(os.path.join(path, 'iterations/discriminator_last'))

    def load_iteration(self, path, iteration):
        self.discriminator.load_weights(os.path.join(path, 'iterations/discriminator/discriminator_weights_{}'.format(iteration)))

    @staticmethod
    def get_iterations_list(path):
        return sorted(set([int(x.replace('discriminator_weights_', '').replace('.data-00000-of-00001', '').replace('.index', ''))
                for x in os.listdir(os.path.join(path, 'iterations/discriminator')) if 'discriminator_weights_' in x]))

    @staticmethod
    def get_timestamp_insert_value(timestamp, max_timestamp):
        assert 0 <= timestamp <= max_timestamp
        return 1 - timestamp / max_timestamp

    @staticmethod
    def get_column_with_timestamps(max_timestamp, dtype=np.float64):
        column = np.empty((max_timestamp, 1), dtype=dtype)
        for i in range(max_timestamp):
            column[i, 0] = AIRL.get_timestamp_insert_value(i+1, max_timestamp)
        return column

    @staticmethod
    def insert_timestamps(paths, copy_paths=False, round_to=None):
        if copy_paths:
            filled_paths = copy.deepcopy(paths)
        else:
            filled_paths = paths
        for i, path in enumerate(filled_paths):
            if 'timestamp_inserted' in path['env_infos']:
                if path['env_infos']['timestamp_inserted'][0]:
                    continue
            obs = path['observations']
            filled_obs = np.c_[obs, AIRL.get_column_with_timestamps(obs.shape[0], dtype=obs.dtype)]
            if round_to is not None:
                def round_nearest(x, a):
                    return np.around(x / a) * a
                filled_obs[:, -1] = round_nearest(filled_obs[:, -1], round_to)
            filled_paths[i]['observations'] = filled_obs
            filled_paths[i]['env_infos']['timestamp_inserted'] = np.full(shape=filled_obs.shape[0], fill_value=True)
        return filled_paths

    @staticmethod
    def insert_timestamp(observation, timestamp, max_timestamp):
        filled_observation = np.append(observation, AIRL.get_timestamp_insert_value(timestamp, max_timestamp))
        return filled_observation
