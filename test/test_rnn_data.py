import numpy as np
import pytest
from rnn import data

@pytest.fixture
def conf():
    _conf = data.Config()
    _conf.time_steps = 2
    _conf.n_output_dim = 2
    return _conf


def test_create_feed(conf):
    x, y = data.create_feed(np.array([1,2,3,4,5,6]),
                            conf)

    np.testing.assert_array_equal(y,
                                  np.array([[3,4],
                                            [4,5],
                                            [5,6],
                                           ]))
    np.testing.assert_array_equal(np.asarray(x),
                                  np.array([[1,2],
                                            [2,3],
                                            [3,4],
                                           ]))

def test_create_labelled_feed(conf):
    x, y = data.create_labelled_feed(np.array([1,2,3,4,5,6]),
                                     np.array([1,1,1,0,0,0]),
                                     conf)
    np.testing.assert_array_equal(y, [1, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(x),
                                  np.array([[1,2],
                                            [2,3],
                                            [3,4],
                                            [4,5],
                                           ]))
