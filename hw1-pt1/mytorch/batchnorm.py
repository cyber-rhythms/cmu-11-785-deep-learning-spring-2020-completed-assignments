# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

# My implementation of the BatchNorm module as part of CMU 11-785 Spring 2020 HW1pt1 assignment.
# The implementation follows that of Ioffe and Szegedy (2015); with a very minor modification.
# The rubric of the HW1pt1 PDF Assignment, under "Appendix C: Batch Normalisation", does not make a bias correction of
# (m / (m - 1)) when variance from training is used to update the running variance that is eventually used for
# inference. The original paper DOES make this correction.

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested.
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # Inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """
        
        self.x = x
        
        if eval == False:
            
            self.mean = np.mean(self.x, axis=0, keepdims=True)
            self.var = np.var(self.x, axis=0, ddof=0, keepdims=True)
            
            self.x_normalised = (self.x - self.mean) / np.sqrt(self.var + self.eps)
            self.out = (self.gamma * self.x_normalised) + self.beta
            
            self.running_mean = (self.alpha * self.running_mean) + ((1 - self.alpha) * self.mean) 
            self.running_var = (self.alpha * self.running_var) + ((1 - self.alpha) * self.var)
            
            return self.out
            
        else:
            self.x_normalised_inf = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.out = (self.gamma * self.x_normalised_inf) + self.beta
            
            return self.out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        
        batch_size = delta.shape[0]
        
        self.dx_normalised = self.gamma * delta 
        
        self.dbeta = np.sum(delta, axis=0, keepdims=True)
        self.dgamma = np.sum(delta * self.x_normalised, axis=0, keepdims=True)
        
        self.dsigma2 = (-1/2) * np.sum(self.dx_normalised
                                       * (self.x - self.mean) 
                                       * ((self.var + self.eps) ** (-3/2)), axis=0, keepdims=True)
        
        self.dmu = (-np.sum(self.dx_normalised * ((self.var + self.eps) ** (-1/2)), axis=0, keepdims=True)
                    - ((2 / batch_size) * self.dsigma2 * (np.sum(self.x - self.mean, axis=0, keepdims=True))))
        
        self.dx_normalised_dx = (self.var + self.eps) ** (-1/2)
        self.dsigma2_dx = (2 / batch_size) * (self.x - self.mean)
        self.dmu_dx = 1 / batch_size
        
        self.dx = ((self.dx_normalised * self.dx_normalised_dx)
                   + (self.dsigma2 * self.dsigma2_dx)
                   + (self.dmu * self.dmu_dx))

        return self.dx