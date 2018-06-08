import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import onnx

import numpy as np

from onnx_tf.backend import prepare
from onnx import helper

generate_onnx = True
torch.manual_seed(1)

# Test inputs
TEST_INPUTS = autograd.Variable(torch.FloatTensor([[[-0.5525, 0.6355, -0.3968]],[[-0.6571, -1.6428, 0.9803]],[[-0.0421, -0.8206, 0.3133]],[[-1.1352, 0.3773, -0.2824]],[[-2.5667, -1.4303, 0.5009]]]))
TEST_INPUTS_ASLIST = [torch.FloatTensor([[-0.5525, 0.6355, -0.3968]]),
                      torch.FloatTensor([[-0.6571, -1.6428, 0.9803]]),
                      torch.FloatTensor([[-0.0421, -0.8206, 0.3133]]),
                      torch.FloatTensor([[-1.1352, 0.3773, -0.2824]]),
                      torch.FloatTensor([[-2.5667, -1.4303, 0.5009]])]

TEST_INPUTS_2 = autograd.Variable(torch.FloatTensor([[[-0.1658, 0.0353, -0.7295]],[[0.2575, -0.2657, -1.7373]],[[0.7332, 1.1558, 0.6375]]]))


# Initial states (unidirectional)
TEST_INITIAL_H = autograd.Variable(torch.FloatTensor([[[0.5438, -0.4057,  1.1341]]]))
TEST_INITIAL_C = autograd.Variable(torch.FloatTensor([[[-1.1115, 0.3501, -0.7703]]]))

# Initial states (bidirectional)
TEST_INITIAL_H_2 = autograd.Variable(torch.FloatTensor([[[0.4975,  0.2355, -1.6301]],
                                                       [[-0.2330,  0.6485, -0.0955]]]))
TEST_INITIAL_C_2 = autograd.Variable(torch.FloatTensor([[[-0.7467,  0.3893,  1.3873]],
                                                        [[ 0.7035, -1.7967, -0.4481]]]))


# Alternatively, generate inputs on the fly:
# inputs = [autograd.Variable(torch.randn((1, 3))) for _ in range(5)]
# hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn((1, 1, 3))))



if generate_onnx:
    # generate an LSTM and save it to lstm.onnx
    lstm = nn.LSTM(3, 3) # Input dim is 3, hidden dim is 3
    #lstm = nn.LSTM(3, 3, bidirectional=True)  
    
    # "Loop style":
    #hidden = (TEST_INITIAL_H, TEST_INITIAL_C) # initialize the hidden state.
    #for i in TEST_INPUTS_ASLIST:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
    #    out, hidden = lstm(i.view(1, 1, -1), hidden)
    #    print(out)


    # "Unroll style":
    # alternatively, we can do the entire sequence all at once.
    # the first value returned by LSTM is all of the hidden states throughout
    # the sequence. the second is just the most recent hidden state
    # (compare the last slice of "out" with "hidden" below, they are the same)
    # The reason for this is that:
    # "out" will give you access to all hidden states in the sequence
    # "hidden" will allow you to continue the sequence and backpropagate,
    # by passing it as an argument  to the lstm at a later time
    # Add the extra 2nd dimension
    out, hidden = lstm(TEST_INPUTS, (TEST_INITIAL_H, TEST_INITIAL_C))
    #out, hidden = lstm(TEST_INPUTS, (TEST_INITIAL_H_2, TEST_INITIAL_C_2))
    print(out)
    print(hidden)


    torch.onnx.export(lstm, (TEST_INPUTS, (TEST_INITIAL_H, TEST_INITIAL_C)), "lstm.onnx", verbose=False)

else:
    # read the model from lstm.onnx with onnx-tensorflow
    model = onnx.load("lstm.onnx")

    tf_rep = prepare(model)

    import tensorflow as tf

    print(tf_rep.run({"0": TEST_INPUTS, "1": TEST_INITIAL_H, "2": TEST_INITIAL_C}))

