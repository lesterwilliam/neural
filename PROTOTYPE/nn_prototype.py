# Import numpy module
import numpy as np
import nn_activation_function as act_func
np.set_printoptions(formatter={'float':'{: 0.4f}'.format})
# Object prototype
class NeuralNetwork:
	def __init__(self, x, y):
		self.input      = x
		self.layer1size = 8
		self.layer2size = 4
		self.weights1   = np.random.rand(self.input.shape[1],self.layer1size)
		self.weights2   = np.random.rand(self.layer1size,self.layer2size)
		self.weights3   = np.random.rand(self.layer2size,1)
		self.y          = y
		self.output     = np.zeros(self.y.shape)
	
	def feedforward(self):
		self.layer1 = act_func.av_f(np.dot(self.input, self.weights1))
		self.layer2 = act_func.av_f(np.dot(self.layer1, self.weights2))
		self.output = act_func.av_f(np.dot(self.layer2, self.weights3))
	
	def backprop(self):
		d_weights3 = np.dot(self.layer2.T, (2*(self.y - self.output) * act_func.av_f_d(self.output)))
		d_weights2 = np.dot(self.layer1.T, np.dot(2*(self.y - self.output) * act_func.av_f_d(self.output), self.weights3.T) * act_func.av_f_d(self.layer2))
		d_weights1 = np.dot(self.input.T, np.dot(np.dot(2*(self.y - self.output) * act_func.av_f_d(self.output), self.weights3.T) * act_func.av_f_d(self.layer2), self.weights2.T) * act_func.av_f_d(self.layer1))
		self.weights1 += d_weights1
		self.weights2 += d_weights2
		self.weights3 += d_weights3
	
	def run(self, personCount, iteration):
		population = np.zeros((personCount,4,8,8))
		for person in range(personCount):
			nn.__init__(X,y)
			for i in range(iteration):
				self.feedforward()
				self.backprop()
			population[person] = self.exportGenes()
		return population
	
	def exportGenes(self):
		export = np.zeros((4,8,8))
		export[0] = nn.weights1
		export[1] = np.pad(nn.weights2, ((0,0),(0,4)), mode='constant', constant_values=0)
		export[2] = np.pad(nn.weights3, ((0,4),(0,7)), mode='constant', constant_values=0)
		export[3][0][0] = self.fitness()
		return (export)
	
	def calc(self, input, genes):
		layer1 = act_func.av_f(np.dot(input, genes[0]))
		layer2 = act_func.av_f(np.dot(layer1, genes[1]))
		output = act_func.av_f(np.dot(layer2, genes[2]))
		return (output[0])
	
	def fitness(self):
		fault = 0
		for item in range(len(self.output)):
			fault += abs(self.output[item] - self.y[item])
		fitness = len(self.output)/fault
		return fitness
	
	def sort(self, population, personCount, iterations):
		for j in range(iterations):
			for i in range(personCount - 1):
				if population[i][3][0][0] < population[i + 1][3][0][0]:
					temp = population[i]
					population[i] = population[i + 1]
					population[i + 1] = temp
		return population
	
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
	personCount = 16
	population = np.zeros((personCount,4,8,8))
	#for person in range(personCount):
	#	nn = NeuralNetwork(X,y)
	#	it_range = 250
	#	for i in range(it_range):
	#		nn.feedforward()
	#		nn.backprop()
	#		progress = float(i) / float(it_range) * 100
	#		if progress - int(progress) == 0:
	#			print("%i%%" %progress)
	#			if i == it_range - 1:
	#				print("Done!\n")
	#	population[person] = nn.exportGenes()
	population = nn.run(16, 1000)
	#for i in range(personCount):
		#print(population[i][0][0][0])
		#print(population[i][3][0][0])
	print(population[0])
	nn.sort(population, 16, 500)
	print("\n")
	print(population[0])
	# Print network data
	
	#print("Input:\n" + str(nn.input) + "\n\n" + str(nn.y) + "\n")
	#print("Weights1:\n" + str(nn.weights1) + "\n")
	#print("Layer1:\n" + str(nn.layer1) + "\n")
	#print("Weights2:\n" + str(nn.weights2) + "\n")
	#print("Layer2:\n" + str(nn.layer2) + "\n")
	#print("Weights3:\n" + str(nn.weights3) + "\n")
	#print("Output:\n" + str(nn.output) + "\n")