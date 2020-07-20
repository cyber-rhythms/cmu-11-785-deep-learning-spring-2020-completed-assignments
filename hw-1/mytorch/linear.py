# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed).

# My implementation of the Linear module as part of the CMU 11-785 Spring 2020 HW1pt1 assignment.
# The derivative of the output of the affine combination to the weight matrix and bias vectors are stored, for use
# during back-propgation.
# Important to remember, as stated in rubric of HW1pt1 PDF, that:
# Computation of the gradient of the loss function with respect to the parameters i.e.
# weights matrix and bias vector occurs here.
# This is averaged over the batch. See the hw1.py script for info for statistical reasons for averaging over batch.

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.momentum_W = np.zeros(self.W.shape)
        self.momentum_b = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        affine_output = np.matmul(x, self.W) + self.b
        self.state = x
        
        return affine_output

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        batch_size = delta.shape[0]
        self.dW = (1 / batch_size) * np.matmul(self.state.T, delta)
        self.db = np.mean(delta, axis=0)
        self.db = self.db[np.newaxis, :]
        
        dx = np.matmul(delta, self.W.T)
        
        return dx