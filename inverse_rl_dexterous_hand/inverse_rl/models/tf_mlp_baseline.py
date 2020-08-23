# Based on https://github.com/justinjfu/inverse_rl

from os import environ
# Select GPU 0 only
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['CUDA_VISIBLE_DEVICES']='0'
environ['MKL_THREADING_LAYER']='GNU'

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from inverse_rl_dexterous_hand.inverse_rl.models.architectures import relu_net
from torch.autograd import Variable
from mjrl.utils.optimize_model import fit_data


class TFMLPBaseline(Model):
    def __init__(self, env_spec, inp_dim=None, inp='obs', learn_rate=1e-3, reg_coef=0.0, batch_size=64, epochs=2,
                 d_hidden=128):
        super(TFMLPBaseline, self).__init__()
        self.n = inp_dim if inp_dim is not None else env_spec.observation_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_coef = reg_coef
        self.inp = inp

        self.model = relu_net(self.n, d_hidden=d_hidden)

        # self.model = nn.Sequential()
        # self.model.add_module('fc_0', nn.Linear(self.n+4, 128))
        # self.model.add_module('relu_0', nn.ReLU())
        # self.model.add_module('fc_1', nn.Linear(128, 128))
        # self.model.add_module('relu_1', nn.ReLU())
        # self.model.add_module('fc_2', nn.Linear(128, 1))

        self.optimizer = tf.optimizers.Adam(self.model.parameters(), lr=learn_rate, weight_decay=reg_coef)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def _features(self, paths):
        if self.inp == 'env_features':
            o = np.concatenate([path["env_infos"]["env_features"][0] for path in paths])
        else:
            o = np.concatenate([path["observations"] for path in paths])
        o = np.clip(o, -10, 10)/10.0
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        N, n = o.shape
        num_feat = int( n + 4 )            # linear + time till pow 4
        feat_mat =  np.ones((N, num_feat)) # memory allocation

        # linear features
        feat_mat[:,:n] = o

        k = 0  # start from this row
        for i in range(len(paths)):
            l = len(paths[i]["rewards"])
            al = np.arange(l)/1000.0
            for j in range(4):
                feat_mat[k:k+l, -4+j] = al**(j+1)
            k += l
        return feat_mat

    def fit(self, paths, return_errors=False):
        featmat = self._features(paths)
        returns = np.concatenate([path["returns"] for path in paths]).reshape(-1, 1)
        featmat = featmat.astype('float32')
        returns = returns.astype('float32')
        num_samples = returns.shape[0]

        # Make variables with the above data
        featmat_var = tf.convert_to_tensor(featmat)
        returns_var = tf.convert_to_tensor(returns)

        if return_errors:
            predictions = np.array(self.model(featmat_var)).ravel()
            errors = returns.ravel() - predictions
            error_before = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)

        with tf.GradientTape() as tape:
            predictions = self.model(featmat_var)
            loss = self.loss_function(returns_var, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if return_errors:
            predictions = np.array(self.model(featmat_var)).ravel()
            errors = returns.ravel() - predictions
            error_after = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)
            return error_before, error_after

    def call(self, path, **kwargs):
        featmat = self._features([path]).astype('float32')
        feat_var = tf.convert_to_tensor(featmat)
        prediction = np.array(self.model(feat_var)).ravel()
        return prediction
