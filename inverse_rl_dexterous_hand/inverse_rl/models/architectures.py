# Based on https://github.com/justinjfu/inverse_rl

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class relu_net(Model):
    def __init__(self, din, layers=2, dout=1, d_hidden=32, initialisation_value=0):
        super(relu_net, self).__init__()
        self.fc_input = Dense(d_hidden, input_shape=(din,), activation='relu')
        self.hidden_layers = [Dense(d_hidden, input_shape=(d_hidden,), activation='relu') for _ in range(layers - 1)]
        self.fc_output = Dense(dout, input_shape=(d_hidden,))
        self.initialisation_value = initialisation_value

    def call(self, x, **kwargs):
        x = self.fc_input(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc_output(x) + self.initialisation_value
        return x


class normalized_relu_net(Model):
    def __init__(self, din, layers=2, dout=1, d_hidden=32, initialisation_value=0):
        super(normalized_relu_net, self).__init__()
        self.fc_input = Dense(d_hidden, input_shape=(din,), activation='relu')
        self.hidden_layers = [Dense(d_hidden, input_shape=(d_hidden,), activation='relu') for _ in range(layers - 1)]
        self.fc_output = Dense(dout, input_shape=(d_hidden,))
        # TODO: std_dev and mean loading from checkpoint does not work
        self.std_dev = tf.Variable(1, dtype=float)
        self.mean = tf.Variable(0, dtype=float)
        self.initialisation_value = initialisation_value

    def call(self, x, **kwargs):
        x = self.fc_input(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc_output(x) + self.initialisation_value
        # normalization will be done, until normalize=False is given
        if 'normalize' in kwargs:
            if kwargs['normalize']:
                x = (x - self.mean.numpy()) / self.std_dev.numpy()
        else:
            x = (x - self.mean.numpy()) / self.std_dev.numpy()
        return x


def feedforward_energy(obs_act, ff_arch=relu_net):
    # for trajectories, using feedforward nets rather than RNNs
    dimOU = int(obs_act.get_shape()[2])
    orig_shape = tf.shape(obs_act)

    obs_act = tf.reshape(obs_act, [-1, dimOU])
    outputs = ff_arch(obs_act)
    dOut = int(outputs.get_shape()[-1])

    new_shape = tf.stack([orig_shape[0],orig_shape[1], dOut])
    outputs = tf.reshape(outputs, new_shape)
    return outputs

