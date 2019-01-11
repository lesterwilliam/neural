# This file contains the basic functions for genetic reproduction

# Import libraries
import random
import numpy as np

# Chance to mutate
mutationChance = 0.1
# Factor a mutation can have compared to the pure input
mutationFactor = 2

# Test data:
#pure = [[0,0,0],
#		[0,0,0],
#		[0,0,0]]

parentA = [[[1.0,1.0,1.0,1.0],
		[0.0,0.0,0.0,0.0],
		[0.0,0.0,0.0,0.0],
		[0.0,0.0,0.0,0.0]],[[1.0,1.0,1.0,1.0],
		[0.0,0.0,0.0,0.0],
		[0.0,0.0,0.0,0.0],
		[0.0,0.0,0.0,0.0]]]

parentB = [[[0.0,0.0,0.0,0.0],
		[0.0,0.0,0.0,0.0],
		[0.0,0.0,0.0,0.0],
		[0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0],
		[0.0,0.0,0.0,0.0],
		[0.0,0.0,0.0,0.0],
		[0.0,0.0,0.0,0.0]]]

# Mixes genes from parents and returns created child
def breedChild(parentA, parentB):
	child = np.full_like(parentA, 0)
	print(np.shape(child)[0])
	print(np.shape(child)[1])
	print(np.shape(child)[2])
	for h in range(np.shape(child)[0]):
		for i in range(np.shape(child)[1]):
			for j in range(np.shape(child)[2]):
				if (int(100 * random.random()) < 50):
					child[h][i][j] = parentA[h][i][j]
				else:
					child[h][i][j] = parentB[h][i][j]
	return child

# Creates a random binary mutation and returns new mutated gene
def mutate_binary(pure, mutationChance):
	mutant = np.zeros((len(pure),1))
	for gene in range(len(pure)):
		if (100 * random.random() < (mutationChance * 100)):
			if pure[gene] == 0:
				mutant[gene] = 1
			else:
				pure[gene] = 0
		else:
			mutant[gene] = pure[gene]
	return mutant

# Creates a random floating mutation and returns new mutated gene
def mutate_floating(pure, mutationChance, mutationFactor):
	mutant = np.copy(pure)
	for i in range(np.shape(mutant)[0]):
		for j in range(np.shape(mutant)[1]):
			if (100 * random.random() < (mutationChance * 100)):
				mutant[i,j] = pure[i,j] * (random.uniform(-1.0*(mutationFactor),1.0*(mutationFactor)))
				if (mutant[i,j] == 0):
					mutant[i,j] = random.uniform(-1,1)
	return mutant

breedChild(parentA,parentB)

#print (mutate_floating(breedChild(parentA, parentB), mutationChance, mutationFactor))