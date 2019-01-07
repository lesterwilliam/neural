import numpy as np

# Activation function: Gradient descent
def av_f(x):
	return sgmd(x)
def av_f_d(x):
	return sgmd_d(x)
	
# Sigmoidal function
def sgmd(x):
	return 1.0/(1+ np.exp(-x))
def sgmd_d(x):
	return x * (1.0 - x)
# Tangens hyperbolicus function
def tanh(x):
	return np.tanh(x)
def tanh_d(x):
	return (1.0 - np.tanh(x)**2)