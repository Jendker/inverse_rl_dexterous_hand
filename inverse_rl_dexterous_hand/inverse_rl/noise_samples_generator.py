import copy
import numpy as np


class WelfordMeanVariance:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0

    # For a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def update(self, new_value):
        self.count += 1
        delta = np.ma.subtract(new_value, self.mean).data
        self.mean += delta / self.count
        delta2 = np.ma.subtract(new_value, self.mean).data
        self.M2 += delta * delta2

    # Retrieve the mean, variance and sample variance from an aggregate
    def retrieve(self):
        if self.count < 2:
            return float('nan')
        else:
            return self.mean, self.M2 / self.count


class NoiseSamplesGenerator:
    def __init__(self, samples_percent, env, range_coefficient=None, remember_only_current_min_max=False,
                 fixed_symmetric_bound=None, fixed_upper_bound=None, fixed_lower_bound=None, std_dev_coefficient=None):
        self.samples_percent = samples_percent
        self.range_coefficient = range_coefficient
        self.std_dev_coefficient = std_dev_coefficient
        if fixed_symmetric_bound and fixed_lower_bound is None:
            self.fixed_lower_bound = -fixed_symmetric_bound
            self.fixed_upper_bound = fixed_symmetric_bound
        else:
            if fixed_lower_bound:
                assert fixed_lower_bound is not None and fixed_upper_bound is not None, "Both fixed_lower_bound and fixed_upper_bound have to be defined"
            self.fixed_lower_bound = fixed_lower_bound
            self.fixed_upper_bound = fixed_upper_bound
            if self.std_dev_coefficient is None:
                assert self.range_coefficient is not None, "If fixed bound values are not defined the range coefficient has to be provided"
        self.remember_only_current_min_max = remember_only_current_min_max
        self.max_values = None
        self.min_values = None

        self.horizon = env.horizon

    def extend_array(self, array, target_length, dim):
        target_shape = list(array.shape)
        target_shape[dim] = target_length
        return np.resize(array, target_shape)

    def get_noise_samples(self, paths):
        noise_samples = []
        if self.fixed_lower_bound is None:
            if self.std_dev_coefficient is not None:
                mean_episode_length = round(float(np.mean([path['observations'].shape[0] for path in paths])))

                obs_dim = paths[0]['observations'].shape[1]
                all_obs = np.ma.empty(shape=[len(paths), self.horizon, obs_dim])
                all_obs.mask = True
                if not hasattr(self, 'values'):
                    self.values = []
                for i, path in enumerate(paths):
                    this_obs = path['observations']
                    all_obs[i, :this_obs.shape[0], :this_obs.shape[1]] = this_obs
                self.values.append(all_obs)
                self.values = self.values[-10:]
                values = [item for sublist in self.values for item in sublist]
                mean = np.ma.mean(values, axis=0)
                std_dev = np.ma.std(values, axis=0)
                for _ in range(int(len(paths) * self.samples_percent / 100)):
                    noise_sample = copy.deepcopy(paths[0])
                    noise_sample['observations'] = np.random.normal(mean[:mean_episode_length], std_dev[:mean_episode_length] * self.std_dev_coefficient)
                    noise_sample['actions'] = self.extend_array(noise_sample['actions'], mean_episode_length, 0)
                    noise_sample['agent_infos']['mean'] = self.extend_array(noise_sample['agent_infos']['mean'], mean_episode_length, 0)
                    noise_sample['agent_infos']['log_std'] = self.extend_array(noise_sample['agent_infos']['log_std'], mean_episode_length, 0)
                    # TODO: why does noise_sample['actions'] = np.empty((self.horizon, noise_sample['actions'].shape[1])) produce nan in IRL loss function?
                    noise_samples.append(noise_sample)
                return noise_samples
            else:
                if self.remember_only_current_min_max:
                    self.max_values = None
                    self.min_values = None
                if self.min_values is None or self.max_values is None:
                    self.max_values = np.full(paths[0]['observations'].shape[1], -float('inf'))
                    self.min_values = np.full(paths[0]['observations'].shape[1], float('inf'))
                for path in paths:
                    self.max_values = np.maximum(np.max(path['observations'], axis=0), self.max_values)
                    self.min_values = np.minimum(np.min(path['observations'], axis=0), self.min_values)
                val_range = (self.max_values - self.min_values) / 2
                middle_values = (self.max_values + self.min_values) / 2
                upper_bound = middle_values + val_range * self.range_coefficient
                lower_bound = middle_values - val_range * self.range_coefficient
        else:
            upper_bound = self.fixed_upper_bound
            lower_bound = self.fixed_lower_bound
        # add noise samples
        for _ in range(int(len(paths) * self.samples_percent / 100)):
            noise_sample = copy.deepcopy(paths[0])
            noise_sample['observations'] = np.random.uniform(lower_bound, upper_bound, noise_sample['observations'].shape)
            noise_samples.append(noise_sample)
        return noise_samples
