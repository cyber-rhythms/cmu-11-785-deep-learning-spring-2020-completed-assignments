# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

# My implementation of the Activations module as part of the CMU 11-785 Spring 2020 HW1pt1 assignment.
# The key point to remember is that during a forward pass through the activation, the output is 'cached'.
# The cached output is later used in the computation of the derivative of the activation output with respect to its
# input, during back-propagation.

import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example.

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed
    # to stay the same for AutoLab.

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # Might we need to store something before returning?
        self.state = 1 / (1 + np.exp(-x))
        return self.state

    def derivative(self):
        # Maybe something we need later in here...
        return self.state * (1 - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        return 1 - (self.state ** 2)


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.maximum(0, x)
        return self.state

    def derivative(self):
        return np.where(self.state > 0, 1.0, 0.0)
