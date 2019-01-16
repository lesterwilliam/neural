# Import libraries
from enum import Enum
import numpy as np
import random

X0 = [1, 0,-1,0]
X1 = [0, 1, 2,0]

class NodeType:
	Input   = 0
	Linear  = 1
	Sigmoid = 2
	TanH    = 3
	ReLU    = 4
	Output  = 5

class Node:
	def __init__(self, type, nodeID, layerID):
		self.nodeID = nodeID
		self.layerID = layerID
		self.type = type
		self.value = random.uniform(-1, 1)
		self.sum = 0
		self.output = 0
	
	def forwardfeed(self):
		if self.layerID == 0:
			self.activation()
		if self.layerID > 0:
			self.sum = 0
			for node in range(net.layers[self.layerID - 1].size):
				net.weights[0].value[0][0] = 0
				net.weights[0].value[0][1] = 2
				net.weights[0].value[0][2] = 4
				net.weights[0].value[0][3] = 6
				net.weights[0].value[1][0] = 0
				net.weights[0].value[1][1] = 3
				net.weights[0].value[1][2] = 9
				net.weights[0].value[1][3] = 12
				
				net.weights[1].value[0][0] = 0.1
				net.weights[1].value[0][1] = 2.1
				net.weights[1].value[0][2] = 4.1
				net.weights[1].value[1][0] = 0.1
				net.weights[1].value[1][1] = 3.1
				net.weights[1].value[1][2] = 9.1
				net.weights[1].value[2][0] = 0.1
				net.weights[1].value[2][1] = 3.1
				net.weights[1].value[2][2] = 9.1
				net.weights[1].value[3][0] = 0.1
				net.weights[1].value[3][1] = 3.1
				net.weights[1].value[3][2] = 9.1
				
				net.weights[2].value[0][0] = 0.1
				net.weights[2].value[1][0] = 3.1
				net.weights[2].value[2][0] = 9.1
				
				self.sum += (net.layers[self.layerID - 1].nodes[node].value) * (net.weights[self.layerID - 1].value[node][self.nodeID])
			print(self.sum)
			self.activation()
	
	def activation(self):
		if self.type == 0:
			pass
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
	
class Layer:
	def __init__(self, type, layerID, input):
		self.type = type
		self.size = len(input)
		self.layerID = layerID
		self.input = input
		self.nodes = []
		# Create Nodes
		for nodeID in range(self.size):
			self.nodes.append(Node(self.type, nodeID, self.layerID))
	
	def output(self):
		return (self.input)
	
	def forwardfeed(self):
		for node in range(self.size):
			self.nodes[node].forwardfeed()

class Weights:
	def __init__(self, layerID, value):
		self.layerID = layerID
		self.value = value

class Network:
	def __init__(self, size):
		self.size = size
		self.layerCount = len(self.size)
		self.layers = []
		self.weights = []
		# Create Layers
		for layerID in range(self.layerCount):
			self.layers.append(Layer(self.size[layerID][1], layerID, [0] * self.size[layerID][0]))
		# Create Weights
		for layerID in range(self.layerCount - 1):
			self.weights.append(Weights(layerID, [[0 for x in range(self.size[layerID+1][0])] for y in range(self.size[layerID][0])]))
	
	def forwardfeed(self):
		for layer in range(self.layerCount):
			self.layers[layer].forwardfeed()

if __name__ == "__main__":
	net = Network([[2,0],[4,2],[3,2],[1,2]])
	print("Layers:")
	for layer in range(net.layerCount):
		for node in range(net.size[layer][0]):
			print(net.layers[layer].nodes[node].value)
		print("\n")
	print("Weights:")
	for layer in range(net.layerCount-1):
		print(net.weights[layer].value)

	#print (net.layers[2].nodes[0].value)
	net.forwardfeed()