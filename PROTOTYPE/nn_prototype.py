# Import numpy module
import numpy as np
import nn_activation_function as act_func
import random
np.set_printoptions(formatter={'float':'{: 0.1f}'.format})
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
		return self.sort(population, 16, 100)
	
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
			for i in range((personCount - 1)):
				if population[i][3][0][0] < population[i + 1][3][0][0]:
					for j in range(np.shape(population)[1]):
						for k in range(np.shape(population)[2]):
							for l in range(np.shape(population)[3]):
								temp = population[i][j][k][l]
								population[i][j][k][l] = population[i+1][j][k][l]
								population[i+1][j][k][l] = temp
		return population
	
	def breedChild(self, parentA, parentB):
		child = np.full_like(parentA, 0)
		for h in range(np.shape(child)[0]):
			for i in range(np.shape(child)[1]):
				for j in range(np.shape(child)[2]):
					if (int(100 * random.random()) < 50):
						child[h][i][j] = parentA[h][i][j]
					else:
						child[h][i][j] = parentB[h][i][j]
		return self.mutate_floating(child, 0.2, 1)
	
	def mutate_floating(self, pure, mutationChance, mutationFactor):
		mutant = np.copy(pure)
		for h in range(np.shape(mutant)[0]):
			for i in range(np.shape(mutant)[1]):
				for j in range(np.shape(mutant)[2]):
					if (100 * random.random() < (mutationChance * 100)):
						mutant[h][i][j] = pure[h][i][j] * (random.uniform(-1.0*(mutationFactor),1.0*(mutationFactor)))
						if (mutant[h][i][j] == 0):
							mutant[h][i][j] = random.uniform(-1,1)
		mutant[3][0][0] = pure[3][0][0]
		return mutant
	
	def reproduce(self, population):
		adam = population[0]
		eva = population[1]
		for i in range(np.shape(population)[0]):
			population[i] = self.breedChild(adam, eva)
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
	
	# Create population of X people and train them Y times. Returns genes of the entire population.
	#population = nn.run(16, 1000)
	for i in range(500):
		population = nn.reproduce(nn.run(16, 100))
	print(population[0][3][0][0])