# Thanks to https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9

import random
import numpy as np

# Mixes genes from parents and returns created child
def breedChild(parentA, parentB):
	if len(parentA) != len(parentB):
		return 0
	child = np.zeros((len(parentA),1))
	for item in range(len(parentA)):
		if (int(100 * random.random()) < 50):
			child[item] = parentA[item]
		else:
			child[item] = parentB[item]
	return child

# Creates a random mutation and returns new mutated gene
def mutate(pure, mutationFactor):
	mutant = np.zeros((len(pure),1))
	for gene in range(len(pure)):
		if (100 * random.random() < (mutationFactor * 100)):
			if pure[gene] == 0:
				mutant[gene] = 1
			else:
				pure[gene] = 0
		else:
			mutant[gene] = pure[gene]
	return mutant