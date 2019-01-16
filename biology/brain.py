# Import libraries
from enum import Enum
import numpy as np

X0 = [1, 0,-1,0]
X1 = [0, 1, 2,0]

class ActFunc:
	Linear  = 0
	Sigmoid = 1
	TanH    = 2

class Node:
	def __init__(self, type, nodeID, layerID, input):
		self.nodeID = nodeID
		self.layerID = layerID
		self.type = type
		self.input = input
		self.sum = sum(self.input)
		self.output = 0
	
	def forwardfeed(self):
		if self.layerID == 0:
			self.activation()
		if self.layerID > 0:
			self.input = (layer[self.layerID - 1] * weights[self.layerID - 1])
			self.activation()
	
	def activation(self):
		if self.type == 0:
			self.output = self.sum
		if self.type == 1:
			self.output =  1.0/(1+ np.exp(-(self.sum)))
		if self.type == 2:
			self.output =  np.tanh(self.sum)
	
class Layer:
	def __init__(self, type, layerID, input):
		self.type = type
		self.size = len(input)
		self.layerID = layerID
		self.input = input
	
	def output(self):
		return (self.input)

class Weights:
	def __init__(self, layerID, input):
		self.layerID = layerID
		self.input = input

class Network:
	def __init__(self, size):
		self.size = size
		self.layerCount = len(size)
		self.layers = []
		for layerID in range(self.layerCount):
			self.layerSize = [0] * self.size[layerID]
			self.layers.append(Layer(0, layerID, self.layerSize))

if __name__ == "__main__":
	net = Network([32,16,16,16,16,16,8,4,2,1])
	for layer in range(net.layerCount):
		print(net.layers[layer].input)