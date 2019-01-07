# Import numpy module
import numpy as np
import nn_activation_function

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
	return 1 - np.tanh(x)

# Object prototype
class NeuralNetwork:
	def __init__(self, x, y):
		self.input      = x
		self.layer1size = 2
		self.layer2size = 2
		self.weights1   = np.random.rand(self.input.shape[1],self.layer1size)
		self.weights2   = np.random.rand(self.layer1size,self.layer2size)
		self.weights3   = np.random.rand(self.layer2size,1)
		self.y          = y
		self.output     = np.zeros(self.y.shape)
		
	def feedforward(self):
		self.layer1 = av_f(np.dot(self.input, self.weights1))
		self.layer2 = av_f(np.dot(self.layer1, self.weights2))
		self.output = av_f(np.dot(self.layer2, self.weights3))
	
	def backprop(self):
		d_weights3 = np.dot(self.layer2.T, (2*(self.y - self.output) * av_f_d(self.output)))
		d_weights2 = np.dot(self.layer1.T, np.dot(2*(self.y - self.output) * av_f_d(self.output), self.weights3.T) * av_f_d(self.layer2))
		d_weights1 = np.dot(self.input.T, np.dot(np.dot(2*(self.y - self.output) * av_f_d(self.output),
		self.weights3.T) * av_f_d(self.layer2), self.weights2.T) * av_f_d(self.layer1))
		self.weights1 += d_weights1
		self.weights2 += d_weights2
		self.weights3 += d_weights3

# Main loop
if __name__ == "__main__":
	# Learning data set
	# Inputs:
	X = np.array([
		[1,1,0,0,0,1,1,0],
		[1,0,1,1,0,0,1,1],
		[1,1,0,0,0,1,1,0],
		[1,0,1,1,0,1,1,0],
		[0,0,1,0,0,1,0,1],
		[0,1,0,1,0,1,1,0],
		[0,0,1,1,1,1,0,1],
		[0,1,0,1,0,1,1,1]])
	# Outputs:
	y = np.array([
		[0],
		[1],
		[0],
		[0],
		[1],
		[1],
		[1],
		[0]])
		
	# Create network object
	nn = NeuralNetwork(X,y)
	
	# Learn data set
	it_range = 10000
	for i in range(it_range):
		nn.feedforward()
		nn.backprop()
		progress = float(i) / float(it_range) * 100
		if progress - int(progress) == 0:
			print("%i%%" %progress)
			if i == it_range - 1:
				print("Done!\n")
	
	# Print network data
	print("Input:\n" + str(nn.input) + "\n\n" + str(nn.y) + "\n")
	print("Weights1:\n" + str(nn.weights1) + "\n")
	print("Layer1:\n" + str(nn.layer1) + "\n")
	print("Weights2:\n" + str(nn.weights2) + "\n")
	print("Layer2:\n" + str(nn.layer2) + "\n")
	print("Weights3:\n" + str(nn.weights3) + "\n")
	print("Output:\n" + str(nn.output) + "\n")
