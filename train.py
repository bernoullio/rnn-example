import csv
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from rnn import data
from rnn.model import Rnn

def main(conf):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.name_scope("train"):
            with tf.variable_scope("model", reuse=None):
                m = Rnn(conf)
        with tf.name_scope("validation"):
            with tf.variable_scope("model", reuse=True):
                m_validate = Rnn(conf)

        tvars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(conf.lr)
        minimize = optimizer.minimize(m.loss,
                           var_list=tvars)
        init = tf.global_variables_initializer()

        all_data = data.load_file(conf.data_file)
        test_index = int(len(all_data) * 0.8)
        if hasattr(conf, 'label_file'):
            labels = data.load_file(conf.label_file)
            x, y = data.create_labelled_feed(all_data[0:test_index], labels[0:test_index], conf)
            x_test, y_test = data.create_labelled_feed(all_data[test_index:], labels[test_index:], conf)
        else:
            x, y = data.create_feed(all_data[0:test_index], conf)
            x_test, y_test = data.create_feed(all_data[test_index:], conf)

        with tf.Session() as sess:
            sess.run(init)
            for i in range(conf.epochs_count):
                if i % 50 == 0:
                    print(".", end="")
                sess.run(minimize,
                         feed_dict={m._inputs: x,
                                    m._targets: y})
            predictions = sess.run(m_validate.predictions,
                                   feed_dict={m_validate._inputs: x_test,
                                              m_validate._targets: y_test})
        pairs = np.asarray([y_test, predictions]).transpose()

        with open('prediction.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            pytest.set_trace()
            for pair in pairs:
                [writer.writerow(p) for p in pair]
                writer.writerow("*****")
        mse = ((predictions - y_test) ** 2 ).mean()
        print(mse)


if __name__ == "__main__":
    main(data.load_conf("fixtures/config.yaml"))
