# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

# My implementation of the Loss module as part of the CMU 11-785 Spring 2020 HW1pt1 assignment.
# Key points to remember:
# In this case, the loss function (cross-entropy) and what is normally the final activation of the neural network
# for multi-class classification (softmax), is bundled together into one function. Was initially confusing.
# That is a code implementation choice, that is also used in PyTorch.
# Implementation of a numerically stable version of both the softmax and log-softmax function.
# See the notes from cs231n and the feedly blog below if you can't remember what is going on:
# https://blog.feedly.com/tricks-of-the-trade-logsumexp/
# https://cs231n.github.io/linear-classify/


import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y

        stability_constant = np.max(x, axis=1)
        stability_constant = stability_constant[:, np.newaxis]
        log_sum_exp = np.log(np.sum(np.exp(x - stability_constant), axis=1))
        log_sum_exp = log_sum_exp[:, np.newaxis]
        
        log_softmax = x - log_sum_exp - stability_constant
        cross_entropy_error = -np.sum(np.multiply(y, log_softmax), axis=1)
        
        self.state = np.exp(log_softmax) - y
        
        return cross_entropy_error

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        return self.state
