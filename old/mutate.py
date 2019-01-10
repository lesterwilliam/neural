# Prototype

import random
import numpy as np

# test data
#pure = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
#mutationFactor = 0.1

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

#print (mutate(pure, mutationFactor))