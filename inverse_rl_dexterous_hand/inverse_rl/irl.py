import logging
logging.disable(logging.CRITICAL)

# utility functions
from mjrl.utils.logger import DataLog

# Import Algs
import numpy as np


def irl_training_class(base):
    class IRL(base):
        def __init__(self, env, policy, baseline,
                     demo_paths=None,
                     seed=123,
                     save_logs=False,
                     irl_model_wt=1.0,
                     irl_batch_size=128,
                     train_irl=True,
                     irl_model=None,
                     no_reward=True,
                     discrim_train_itrs=20,
                     entropy_weight=0.1,
                     augmentation=False,
                     lower_lr_on_main_loop_percentage=None,
                     discr_lr=1e-3,
                     call_super=False,
                     **kwargs
                     ):

            super().__init__(env=env, policy=policy, baseline=baseline, demo_paths=demo_paths,
                             save_logs=save_logs, **kwargs)
            self.env = env
            self.policy = policy
            self.baseline = baseline
            self.seed = seed
            self.save_logs = save_logs
            self.running_score = None
            self.demo_paths = demo_paths
            self.iter_count = 0.0
            self.irl_model = irl_model
            self.irl_model_wt = irl_model_wt
            self.irl_batch_size = irl_batch_size
            self.train_irl = train_irl
            self.no_reward = no_reward
            self.entropy_weight = entropy_weight
            self.discrim_train_itrs = discrim_train_itrs
            self.global_status = dict()
            self.dump_paths = False
            self.default_lr = discr_lr
            self.lower_lr_on_main_loop_percentage = lower_lr_on_main_loop_percentage
            if isinstance(self.lower_lr_on_main_loop_percentage, list):
                self.lower_lr_on_main_loop_percentage = np.array(self.lower_lr_on_main_loop_percentage)
            if augmentation > 0:
                from inverse_rl_dexterous_hand.inverse_rl.augmentation import Augmentation
                self.augmentation = Augmentation(env, augment_times=augmentation)
            else:
                self.augmentation = None
            if save_logs: self.logger = DataLog()

        @property
        def checkpoint(self):
            save_checkpoint_funct = getattr(self.irl_model, "save_checkpoint", None)
            if not save_checkpoint_funct:
                return [self.policy, self.baseline, self.irl_model, self.global_status]
            else:
                return [self.policy, self.baseline, self.global_status]

        def save_checkpoint(self, **kwargs):
            save_checkpoint_funct = getattr(self.irl_model, "save_checkpoint", None)
            if save_checkpoint_funct:
                save_checkpoint_funct(kwargs['path'], kwargs['iteration'])

        def load_checkpoint(self, checkpoint, **kwargs):
            load_checkpoint_funct = getattr(self.irl_model, "load_checkpoint", None)
            if load_checkpoint_funct:
                load_checkpoint_funct(kwargs['path'])
                self.policy, self.baseline, self.global_status = checkpoint
            else:
                self.policy, self.baseline, self.irl_model, self.global_status = checkpoint

        def eval_irl(self, paths, training_paths_from_policy=True):
            if self.no_reward:
                tot_rew = 0
                for path in paths:
                    tot_rew += np.sum(path['rewards'])
                    path['rewards'] *= 0
                if training_paths_from_policy:
                    self.logger.log_kv('OriginalTaskAverageReturn', tot_rew / float(len(paths)))

            if self.irl_model_wt <= 0:
                return paths

            probs = self.irl_model.eval(paths)
            probs_flat = np.concatenate(probs)  # trajectory length varies

            if self.train_irl and training_paths_from_policy:
                self.logger.log_kv('IRLRewardMean', np.mean(probs_flat))
                self.logger.log_kv('IRLRewardMax', np.max(probs_flat))
                self.logger.log_kv('IRLRewardMin', np.min(probs_flat))

            if self.irl_model.score_trajectories:
                # TODO: should I add to reward here or after advantage computation? by Justin Fu
                for i, path in enumerate(paths):
                    path['rewards'][-1] += self.irl_model_wt * probs[i]
            else:
                for i, path in enumerate(paths):
                    path['rewards'] += self.irl_model_wt * probs[i]
            return paths

        def fit_irl(self, paths, main_loop_step, main_loop_percentage, num_cpu, policy_updates_count):
            if self.irl_model_wt <= 0 or not self.train_irl:
                return

            if self.no_reward:
                tot_rew = 0
                for path in paths:
                    tot_rew += np.sum(path['rewards'])
                    path['rewards'] *= 0

            lr = self.default_lr
            if self.lower_lr_on_main_loop_percentage is not None:
                elements_lower_than_thresholds = (self.lower_lr_on_main_loop_percentage < main_loop_percentage).sum()
                lr *= 0.1 ** elements_lower_than_thresholds
            mean_loss = self.irl_model.fit(paths, policy=self.policy, logger=self.logger, batch_size=self.irl_batch_size,
                                           max_itrs=self.discrim_train_itrs, lr=lr, num_cpu=num_cpu,
                                           policy_updates_count=policy_updates_count, main_loop_step=main_loop_step,
                                           main_loop_percentage=main_loop_percentage)
            self.logger.log_kv('IRLLoss', mean_loss)
    return IRL
