#
# Copyright 2025 Micha Erkel
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import sys
import os

from torch import index_copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math import ceil, floor
from PINNLearning.models import create_model


def test_create_model():
    out_dim = 1
    inp_dim = 1
    num_layers = 4
    weights_dim = [9, 4, 2, 12]  # should only be a list with length num_layers

    model = create_model(num_layers, weights_dim, out_dim=out_dim)
    model.build(input_shape=(1, inp_dim))

    # Look if the model is visually correct
    print(model.summary())

    # Check for the correct dimensions
    weights = model.get_weights()
    biases = [i for i in range(1, len(weights) - 1) if i % 2]
    for indx in range(1, len(weights) - 1):
        dim = list(weights[indx].shape)

        if indx in biases:
            dim_tar = weights_dim[biases.index(indx)]
            assert dim[0] == dim_tar
        else:
            low_bound = indx - (indx // 2 + 1)
            high_bound = low_bound + 1

            if low_bound == num_layers - 1:
                assert dim == [weights_dim[low_bound], out_dim]
            else:
                assert dim == weights_dim[low_bound: high_bound + 1]

    # Check if num_layers gets properly extendend
    weights_dim = 5  # shold only be a single number

    model2 = create_model(num_layers, weights_dim, out_dim=out_dim)
    model2.build(input_shape=(1, inp_dim))

    weights = model2.get_weights()
    for indx in range(1, len(weights) - 1):
        dim = list(weights[indx].shape)

        if len(dim) == 1:
            dim_tar = weights_dim
            assert dim[0] == dim_tar


if __name__ == "__main__":
    test_create_model()
