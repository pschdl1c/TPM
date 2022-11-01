import numpy as np
from update_rules import *

class TPM:
    def __init__(self, k=6, n=4, l=6):
        self.k = k
        self.n = n
        self.l = l
        self.W = np.random.randint(-l, l + 1, [k, n])

    def get_output(self, X):
        sigma = np.sign(np.sum(X * self.W, axis=1))
        sigma[sigma == 0] = -1
        self.tau = np.prod(sigma)
        self.X = X
        self.sigma = sigma
        return self.tau

    def __call__(self, X):
        return self.get_output(X)

    def update(self, tau2, update_rule='hebbian'):
        if (self.tau == tau2):
            if update_rule == 'hebbian':
                hebbian(self.W, self.X, self.sigma, self.tau, tau2, self.l)
            elif update_rule == 'anti_hebbian':
                anti_hebbian(self.W, self.X, self.sigma, self.tau, tau2, self.l)
            elif update_rule == 'random_walk':
                random_walk(self.W, self.X, self.sigma, self.tau, tau2, self.l)
            else:
                raise Exception("Invalid update rule.")
        return