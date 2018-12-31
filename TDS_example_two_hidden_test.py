# All credits to James Loy, you're the man!
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,4)                 
        self.weights3	= np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))

    def backprop(self):
   		d_weights3 = np.dot(self.layer2.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
		d_weights2 = np.dot(self.layer1.T, np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2))
		d_weights1 = np.dot(self.input.T, np.dot(np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2), self.weights2.T) * sigmoid_derivative(self.layer1))
		self.weights1 += d_weights1
		self.weights2 += d_weights2
		self.weights3 += d_weights3

if __name__ == "__main__":
    X = np.array([[0.1],
                  [0.2],
                  [0.3],
                  [0.4]])
    y = np.array([[0.1],
                  [0.2],
                  [0.3],
                  [0.4]])
    nn = NeuralNetwork(X,y)
    it_range = 100000
    for i in range(it_range):
        nn.feedforward()
        nn.backprop()

        progress = float(i) / float(it_range) * 100
        if progress - int(progress) == 0:
        	print("%i%%" %progress)
        if i == it_range - 1:
        	print("Done!\n")

    
    print("Input:\n" + str(nn.input) + "\n\n" + str(nn.y) + "\n")
    print("Weights1:\n" + str(nn.weights1) + "\n")
    print("Layer1:\n" + str(nn.layer1) + "\n")
    print("Weights2:\n" + str(nn.weights2) + "\n")
    print("Layer2:\n" + str(nn.layer2) + "\n")
    print("Weights3:\n" + str(nn.weights3) + "\n")
    print("Output:\n" + str(nn.output) + "\n")