# prototype

import random
import numpy as np

# test data
#parentA = [0,0,9,0,0,0,0,0,1,5,8,7,9,5]
#parentB = [1,1,1,1,1,1,1,1,5,9,4,8,5,5]

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

def breedChildren(parents, number_of_child):
	nextPopulation = []

print (breedChild(parentA, parentB))