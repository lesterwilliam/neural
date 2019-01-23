# Import libraries
from enum import Enum
import numpy as np
import random
import time

X0 = 0.1
X1 = 0.2
X2 = 0.3
X3 = 0.4

Y0 = [1, 0, 1, 0]

class NodeType:
	Input   = 0
	Linear  = 1
	Sigmoid = 2
	TanH    = 3
	ReLU    = 4
	SoftMax = 5
	Output  = 6

class Node:
	def __init__(self, type, nodeID, layerID, layerCount):
		self.nodeID = nodeID
		self.layerID = layerID
		self.layerCount = layerCount
		self.type = type
		self.value = random.uniform(-1, 1)
		self.sum = 0
		self.output = 0
		self.error = 0
	
	def forwardfeed(self):
		if self.layerID == 0:
			pass
		if self.layerID > 0:
			self.sum = 0
			for node in range(net.layers[self.layerID - 1].size):
				# Test data
				
				self.sum += (net.layers[self.layerID - 1].nodes[node].output) * (net.weights[self.layerID - 1].value[node][self.nodeID])
			#print(self.sum)
			self.activation()
	
	def activation(self):
		if self.type == 0:
			self.output = self.sum
		if self.type == 1:
			self.output = self.sum
		if self.type == 2:
			self.output =  1.0/(1+ np.exp(-(self.sum)))
		if self.type == 3:
			self.output =  np.tanh(self.sum)
		if self.type == 4:
			self.output = self.sum * (self.sum > 0)
		if self.type == 5:
			self.output = self.sum
		if self.type == 6:
			self.output = self.sum
	
	def getError(self):
		# Calculate node error from all nodes in next layer and weights
		pass
		self.error = np.dot(

class Layer:
	def __init__(self, type, layerID, input, layerCount):
		self.type = type
		self.size = len(input)
		self.layerID = layerID
		self.input = input
		self.nodes = []
		self.layerCount = layerCount
		# Create Nodes
		for nodeID in range(self.size):
			self.nodes.append(Node(self.type, nodeID, self.layerID, self.layerCount))
	
	def output(self):
		return (self.input)
	
	def forwardfeed(self):
		for node in range(self.size):
			self.nodes[node].forwardfeed()
	
	def backprop(self):
		for node in range(self.size):
			self.nodes[node].backprop()

class Weights:
	def __init__(self, layerCount, layerID, value):
		self.layerCount = layerCount
		self.layerID = layerID
		self.value = value
	
	def backprop(self):
		# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
		#d_weights3 = np.dot(self.layer2.T, (2*(self.y - self.output) * act_func.av_f_d(self.output)))
		#d_weights2 = np.dot(self.layer1.T, np.dot(2*(self.y - self.output) * act_func.av_f_d(self.output), self.weights3.T) * act_func.av_f_d(self.layer2))
		#d_weights1 = np.dot(self.input.T, np.dot(2*(self.y - self.output) * act_func.av_f_d(self.layer2), self.weights2.T) * act_func.av_f_d(self.layer1))

		for x in range(len(self.value)):
			for y in range(len(self.value[x])):
				if self.layerID == self.layerCount -2:
					# weight algorythm for output weights
					newWeight = self.value[x][y]
					leftLayer = net.layers[self.layerID].nodes[x].output
					rightLayer = net.layers[self.layerID+1].nodes[y].output
					outputLayer = net.layers[self.layerCount-1].nodes[y].output
					soll = Y0[y]
					error = soll - outputLayer
					d_rightLayer = 1.0/(1+ np.exp(-((net.layers[self.layerID+1].nodes[y].output))))
					d_outputLayer = 1.0/(1+ np.exp(-((net.layers[self.layerCount-1].nodes[y].output))))
					
					newWeight += np.dot(leftLayer, (2*error*d_outputLayer))
				else:
					pass
					# weight algorythm for non-output weights
					newWeight = self.value[x][y]
					leftLayer = net.layers[self.layerID].nodes[x].output
					rightLayer = net.layers[self.layerID+1].nodes[y].output
					outputLayer = net.layers[self.layerCount-1].nodes[y].output
					soll = Y0[y]
					error = soll - outputLayer
					d_rightLayer = 1.0/(1+ np.exp(-((net.layers[self.layerID+1].nodes[y].output))))
					d_outputLayer = 1.0/(1+ np.exp(-((net.layers[self.layerCount-1].nodes[y].output))))
					
					newWeight += np.dot(leftLayer, np.dot(2 * error * d_outputLayer, self.weights3.T) * d_rightLayer)
				self.value[x][y] = newWeight
	
class Network:
	def __init__(self, size):
		self.size = size
		self.layerCount = len(self.size)
		self.weightCount = self.layerCount -1
		self.layers = []
		self.weights = []
		self.error = 0
		# Create Layers
		for layerID in range(self.layerCount):
			self.layers.append(Layer(self.size[layerID][1], layerID, [0] * self.size[layerID][0], self.layerCount))
		# Create Weights
		for layerID in range(self.weightCount):
			self.weights.append(Weights(self.layerCount, layerID, [[random.uniform(-1,1) for x in range(self.size[layerID+1][0])] for y in range(self.size[layerID][0])]))
	
	def forwardfeed(self):
		for layer in range(self.layerCount):
			self.layers[layer].forwardfeed()
		self.error = 0
		for nodes in range(self.size[self.weightCount][0]):
			self.error += ((Y0[nodes] - net.layers[self.weightCount].nodes[nodes].output)**2)/2
	
	def backprop(self):
		for weight in reversed(range(self.weightCount)):
			self.weights[weight].backprop()

def printNet(network):
	print("\nNetwork:")
	for layer in range(network.layerCount):
		print("\nLayer ", layer, ":")
		for node in range(network.size[layer][0]):
			print("Node  ", node, ": ", network.layers[layer].nodes[node].output)
	
def printError(network):
	print("\nError    : ", network.error)
	
def printWeights(network):
	print("\nWeights:")
	for layer in range(network.weightCount):
		print("\nWeight ", layer, ":")
		print(net.weights[layer].value)
	
if __name__ == "__main__":
	startTime = time.time()
	print("Programm started.")
	# Create Network Object
	net = Network([[4,0],[4,1]])
	
	# Set input data
	net.layers[0].nodes[0].output = X0
	net.layers[0].nodes[1].output = X1
	net.layers[0].nodes[2].output = X2
	net.layers[0].nodes[3].output = X3
	printNet(net)
	# Run net
	for i in range(100):
		net.forwardfeed()
		net.backprop()
	net.forwardfeed()
	# Prints
	printNet(net)
	printError(net)
	printWeights(net)
	
	# Print time used for completion of program
	print("\nTotal time used: ", '{:.3f}'.format(time.time()-startTime), "seconds")