import numpy as np

def theta(t1, t2):
	return 1 if t1 == t2 else 0

def hebbian(W, X, sigma, tau1, tau2, l):
	for (i, j), _ in np.ndenumerate(W):
		W[i, j] += X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2)
		W[i, j] = np.clip(W[i, j] , -l, l)
	return W

def anti_hebbian(W, X, sigma, tau1, tau2, l):
	for (i, j), _ in np.ndenumerate(W):
		W[i, j] -= X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2)
		W[i, j] = np.clip(W[i, j], -l, l)

def random_walk(W, X, sigma, tau1, tau2, l):
	for (i, j), _ in np.ndenumerate(W):
		W[i, j] += X[i, j] * theta(sigma[i], tau1) * theta(tau1, tau2)
		W[i, j] = np.clip(W[i, j] , -l, l)