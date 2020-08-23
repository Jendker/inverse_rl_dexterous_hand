# based on paper "Fast Is Better Than Free: Revisiting Adversarial Training"

import tensorflow as tf
import numpy as np


class AdversarialSamples:
    def __init__(self, adversarial_samples_percentage, epsilon=0.03, alpha_ratio=0.25, custom_range=None):
        self.min_obs = None
        self.max_obs = None
        self.min_act = None
        self.max_act = None
        self.adversarial_samples_percentage = adversarial_samples_percentage
        assert 0 < epsilon < 1, "Epsilon has to be between 0 and 1 (ratio of min/max for perturbations)"
        self.epsilon = epsilon
        self.custom_range = custom_range
        self.alpha_ratio = alpha_ratio

    @staticmethod
    def _get_next_states(obs, act, pad_val=0.0):
        nobs = tf.concat([obs[1:, :], pad_val * tf.expand_dims(tf.ones_like(obs[0]), axis=0)], axis=0)
        nact = tf.concat([act[1:, :], pad_val * tf.expand_dims(tf.ones_like(act[0]), axis=0)], axis=0)
        return nobs, nact

    def add_adversarial_samples(self, discriminator, feed_dict, labels, use_timestamp):
        act_batch = feed_dict['act_t']
        obs_batch = feed_dict['obs_t']
        lprobs_batch = feed_dict['lprobs']

        if self.custom_range is not None:
            obs_range = self.custom_range
            act_range = self.custom_range
        else:
            raise NotImplemented("Not implemented step size yet")
            obs_range = (self.max_obs - self.min_obs) * self.epsilon
            act_range = (self.max_act - self.min_act) * self.epsilon

        alpha = self.custom_range * self.alpha_ratio

        adversarial_samples_count = int(self.adversarial_samples_percentage * obs_batch.shape[0])

        if isinstance(obs_batch, np.ndarray):
            obs_batch = tf.convert_to_tensor(obs_batch)
        if isinstance(act_batch, np.ndarray):
            act_batch = tf.convert_to_tensor(act_batch)

        obs_shape = (adversarial_samples_count, obs_batch.shape[1])
        act_shape = (adversarial_samples_count, act_batch.shape[1])

        obs_delta = tf.random.uniform(obs_shape, -obs_range, obs_range)
        act_delta = tf.random.uniform(act_shape, -act_range, act_range)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(act_delta)
            tape.watch(obs_delta)

            input_act = act_batch[:adversarial_samples_count] + act_delta
            input_obs = obs_batch[:adversarial_samples_count] + obs_delta
            input_nobs, _ = self._get_next_states(input_obs, input_act)
            input_lprobs = tf.expand_dims(tf.convert_to_tensor(lprobs_batch[:adversarial_samples_count]), axis=1)
            input_labels = labels[:adversarial_samples_count]

            inputs = [input_act, input_obs, input_nobs, input_lprobs]

            discrim_output, log_p_tau, log_pq, log_q_tau = discriminator(inputs)
            loss = discriminator.calculate_loss(log_p_tau, log_pq, log_q_tau, input_labels)
        obs_delta_grad = tape.gradient(loss, obs_delta)
        act_delta_grad = tape.gradient(loss, act_delta)

        obs_delta += alpha * tf.sign(obs_delta_grad)
        obs_delta = tf.clip_by_value(obs_delta, -obs_range, obs_range)
        if use_timestamp:
            obs_delta = tf.concat([obs_delta[:, :-1], tf.zeros((obs_delta.shape[0], 1), dtype=obs_delta.dtype)], axis=1)

        if act_delta_grad is not None:
            act_delta += alpha * tf.sign(act_delta_grad)
            act_delta = tf.clip_by_value(act_delta, -act_range, act_range)
        else:
            act_delta = tf.zeros_like(act_delta)

        all_obs_batch = tf.concat([obs_delta + obs_batch[:adversarial_samples_count, :], obs_batch[adversarial_samples_count:, :]], axis=0)
        all_act_batch = tf.concat([act_delta + act_batch[:adversarial_samples_count, :], act_batch[adversarial_samples_count:, :]], axis=0)
        all_nobs_batch, all_act_batch = self._get_next_states(all_obs_batch, all_act_batch)

        return all_obs_batch, all_act_batch, all_nobs_batch, all_act_batch
