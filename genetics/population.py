# 

# Import libraries
import random
import numpy as np

class Population:
	def __init__(self, size):
		self.populationSize = size
		self.population = np.zeros((self.size,4,8,8))
	
	def sortFittest(self):
		