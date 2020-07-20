# My implementation of the MLP module as part of CMU 11-785 Spring 2020 HW1pt1 assignment.
"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):
    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Store dimensions which are used to create the Linear and BatchNorm layers.        
        linear_layer_args = [input_size] + hiddens + [output_size]
        bn_layer_args = hiddens + [output_size]

        # Store Linear and BatchNorm layer objects.       
        self.linear_layers = [Linear(linear_layer_args[layer], linear_layer_args[layer + 1],
                                     weight_init_fn, bias_init_fn) for layer in range(self.nlayers)]
        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(bn_layer_args[bn_layer]) for bn_layer in range(num_bn_layers)]

        # Indicate whether batch-norm module is being used in a particular layer.       
        self.bn_indicator = [True if layer == num_bn_layers - 1 else False for layer in range(self.nlayers)]

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Load the data.

        self.prev_layer_out = x

        # Forward propagate a batch of training inputs through successive layers of affine
        # and non-linear transformations.
        # The "BatchNorm" transformation, if used, depends on whether the MLP is in training mode or inference mode.

        for layer in range(self.nlayers):
            self.affine_out = self.linear_layers[layer].forward(self.prev_layer_out)

            if self.bn_indicator[layer]:
                eval_status = not self.train_mode
                self.bn_out = self.bn_layers[layer].forward(self.affine_out, eval_status)
                self.activation_out = self.activations[layer].forward(self.bn_out)
            else:
                self.activation_out = self.activations[layer].forward(self.affine_out)

            if layer == self.nlayers - 1:
                self.output = self.activation_out
            else:
                self.prev_layer_out = self.activation_out

        return self.output

    def zero_grads(self):
        # Resets cached derivatives with respect to parameters in the linear and BatchNorm layers to 0. 
        for layer in range(len(self.linear_layers)):
            self.linear_layers[layer].dW = np.zeros(self.linear_layers[layer].dW.shape)
            self.linear_layers[layer].db = np.zeros(self.linear_layers[layer].db.shape)

            if self.bn_indicator[layer]:
                self.bn_layers[layer].dgamma = np.zeros(self.bn_layers[layer].dgamma.shape)
                self.bn_layers[layer].dbeta = np.zeros(self.bn_layers[layer].dbeta.shape)
            else:
                pass

    def step(self):
        # Updates the parameters (weights and biases) via mini-batch SGD using the gradients computed via
        # back-propagation.
        # Momentum, if used, updates the parameters according to the formula by Polyak (1964), stated in rubric.
        # This uses an exponentially decaying moving average of past gradients.
        # It is NOT the variant known as "Nesterov momentum".
        # If BatchNorm is used, the relevant parameters are also updated via mini-batch SGD.
        # Under mini-batch SGD, parameter updates are made using the gradient of the loss with respect to the parameters
        # for a mini-batch. This is an unbiased estimator of the exact gradient of the generalisation error.
        for layer in range(len(self.linear_layers)):

            # Momentum module
            if self.momentum != 0.0:
                # Compute weights and bias updates in presence of momentum.
                W_update = ((self.momentum * self.linear_layers[layer].momentum_W)
                           - (self.lr * self.linear_layers[layer].dW))
                b_update = ((self.momentum * self.linear_layers[layer].momentum_b)
                           - (self.lr * self.linear_layers[layer].db))

                # Update weights and biases.
                self.linear_layers[layer].W = self.linear_layers[layer].W + W_update
                self.linear_layers[layer].b = self.linear_layers[layer].b + b_update

                # Store current round update for use in next round.
                self.linear_layers[layer].momentum_W = W_update
                self.linear_layers[layer].momentum_b = b_update
            else:
                # Vanilla mini-batch stochastic gradient descent.
                self.linear_layers[layer].W = self.linear_layers[layer].W - self.lr * self.linear_layers[layer].dW
                self.linear_layers[layer].b = self.linear_layers[layer].b - self.lr * self.linear_layers[layer].db

                # BatchNorm parameter update, if used.
                if (self.bn_indicator[layer]):
                    self.bn_layers[layer].gamma = self.bn_layers[layer].gamma - self.lr * self.bn_layers[layer].dgamma
                    self.bn_layers[layer].beta = self.bn_layers[layer].beta - self.lr * self.bn_layers[layer].dbeta
                else:
                    pass

    def backward(self, labels):
        # Conventionally, in NN literature, the final layer of the MLP consists of an affine transformation followed by
        # a final non-linear output activation function (softmax in our context).
        # The outputs are then passed together with training labels to a loss function. 

        # In this implementation, and similar to PyTorch, the final output activation (softmax) and loss (cross-entropy)
        # are combined into a SoftmaxCrossEntropy criterion, and is separate to the main MLP object.
        # The final layer of the MLP object consists of an affine combination followed by an Identity activation. 

        # The derivatives of the loss with respect to final output, affine inputs, activation inputs, BatchNorm inputs
        # are stored as attributes in self.d_<layer-input>.

        self.loss = self.criterion.forward(self.output, labels)
        self.d_output = self.criterion.derivative()

        mini_batch_param_grad_est = []
        bn_param_grad_est = []

        # Backpropagate the derivative of the loss with respect to a batch of final outputs through
        # successive layers of non-linear activations, BatchNorm transformations if used, and affine transformations. 

        self.d_affine_in = self.d_output

        for layer in range(self.nlayers - 1, -1, -1):
            self.d_activation_in = self.d_affine_in * self.activations[layer].derivative()

            if self.bn_indicator[layer]:
                self.d_bn_in = self.bn_layers[layer].backward(self.d_activation_in)
                self.d_affine_in = self.linear_layers[layer].backward(self.d_bn_in)

                bn_weight_grad_est = self.bn_layers[layer].dgamma
                bn_offset_grad_est = self.bn_layers[layer].dbeta

                bn_param_grad_est.append(bn_weight_grad_est)
                bn_param_grad_est.append(bn_offset_grad_est)
            else:
                self.d_affine_in = self.linear_layers[layer].backward(self.d_activation_in)

            weight_grad_est = self.linear_layers[layer].dW
            bias_grad_est = self.linear_layers[layer].db

            mini_batch_param_grad_est.append(weight_grad_est)
            mini_batch_param_grad_est.append(bias_grad_est)

        return mini_batch_param_grad_est, bn_param_grad_est

    def error(self, labels):
        return (np.argmax(self.output, axis=1) != np.argmax(labels, axis=1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Reproducibility seed
    np.random.seed(21)

    for e in range(nepochs):

        # Shuffle training and validation instances in place at the beginning of each epoch.
        # Repeated calls to seed to ensure that shuffling of training instances and validation instances
        # preserves per-instance correspondence between an input and its label.

        shuffle_queue = [trainx, trainy, valx, valy]
        for dataset in shuffle_queue:
            np.random.seed(22)
            np.random.shuffle(dataset)

        mlp.train()             # Place the NN in "train" mode.

        epoch_loss_by_batch = []
        epoch_train_class_error_by_batch = []
        epoch_val_loss_by_batch = []
        epoch_val_class_error_by_batch = []

        for b in range(0, len(trainx), batch_size):
            x_mini_batch = trainx[b:b + batch_size]
            y_mini_batch = trainy[b:b + batch_size]

            mlp.zero_grads()
            mlp.forward(x_mini_batch)
            mlp.backward(y_mini_batch)
            mlp.step()

            # Compute and store average loss and average classification error for each batch.
            # For clarity, these statistics are computed by summing per-instance loss and per-instance classification
            # error over all training instances within the batch, and then dividing by the size of the batch.

            mini_batch_loss = mlp.total_loss(y_mini_batch) / batch_size
            epoch_loss_by_batch.append(mini_batch_loss)

            mini_batch_class_error = mlp.error(y_mini_batch) / y_mini_batch.shape[0]
            epoch_train_class_error_by_batch.append(mini_batch_class_error)

        # Assignment scope is unclear here.
        # We have chosen to report the average loss and average classification error for the FINAL batch of the epoch
        # as statistics to report. This results in the training loss and training curves decreasing 'noisily'.
        # ALTERNATIVE: Compute the average loss and average classification over the entire epoch by summing all the
        # batch average losses and batch classification errors recorded over the entire epoch, and dividing by number of
        # batches in the epoch.This results in the training loss and training curves decreasing 'smoothly'.
        # Still not entirely clear on the mechanics.
        # O/S to return to at a later date - leave a commit message in GitHub.

        training_losses[e] = mini_batch_loss
        training_errors[e] = mini_batch_class_error

        for b in range(0, len(valx), batch_size):
            mlp.eval()          # Place the NN in "validation/test" mode.

            x_mini_batch = valx[b:b + batch_size]
            y_mini_batch = valy[b:b + batch_size]

            mlp.zero_grads()
            mlp.forward(x_mini_batch)

            mini_batch_loss = mlp.total_loss(y_mini_batch) / batch_size
            epoch_val_loss_by_batch.append(mini_batch_loss)

            mini_batch_class_error = mlp.error(y_mini_batch) / y_mini_batch.shape[0]
            epoch_val_class_error_by_batch.append(mini_batch_class_error)

        validation_losses[e] = np.mean(epoch_val_loss_by_batch)
        validation_errors[e] = np.mean(epoch_train_class_error_by_batch)

    return training_losses, training_errors, validation_losses, validation_errors
