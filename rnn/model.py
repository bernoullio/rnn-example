import pytest
import tensorflow as tf


def input_placeholders(conf):
    x = tf.placeholder(tf.float32,
                       [None, conf.time_steps, conf.n_features],
                       name="x")
    y = tf.placeholder(tf.float32,
                       [None, conf.n_output_dim],
                       name="y")
    return x, y


class Rnn(object):
    def __init__(self, conf):
        self._conf = conf
        self._inputs, self._targets = input_placeholders(conf)
        sequential_inputs = split(self._inputs,
                                  self._conf.time_steps,
                                  self._conf.n_features)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_layers(),
                                                   state_is_tuple=True)
        stacked_lstm.zero_state(self._conf.batch_size, tf.float32)
        outputs, states = tf.contrib.rnn.static_rnn(stacked_lstm,
                                                    sequential_inputs,
                                                    dtype=tf.float32)

        output_size = self._conf.rnn_layers[-1]  # the top most rnn layer

        w = tf.get_variable("w",
                            [output_size, self._conf.n_output_dim],
                            initializer=tf.constant_initializer(0),
                            dtype=tf.float32)
        b = tf.get_variable("b",
                            [self._conf.n_output_dim],
                            initializer=tf.constant_initializer(0),
                            dtype=tf.float32)
        self.predictions = tf.matmul(outputs[-1], w) + b

    @property
    def loss(self):
        return tf.reduce_mean(tf.losses.mean_squared_error(self.predictions, self._targets))

    def lstm_layers(self):
        def _wrap(cell):
            return tf.contrib.rnn.DropoutWrapper(cell,
                                                 input_keep_prob=self._conf.rnn_in_keep,
                                                 output_keep_prob=self._conf.rnn_out_keep)
        return [_wrap(tf.contrib.rnn.LSTMBlockCell(n_cells)) for n_cells in self._conf.rnn_layers]

def split(tensor, time_steps, n_features):
    """ inputs is of shape [batch_size, time_steps, n_features]
    """
    x = tf.transpose(tensor, [1, 0, 2])  # permute batch_size and time_steps
    x = tf.reshape(x, [-1, n_features])  # Flatten into (time_steps * batch_size, n_features)
    x = tf.split(x, time_steps, 0)  # split into a list  of #(time_steps) tensors of shape (batch_size, n_features)
    return x
