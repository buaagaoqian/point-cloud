
import numpy as np

class Scene:
	def __init__(self):
		self.index = 0
		

	def __iter__(self):
		self.index = 0
		return self

	def __next__(self):
		if self.index == 10:
			raise StopIteration()
		self.index += 1
		sample = np.random.rand(1024, 8)
		ground_truth = np.random.rand(1024, 8)
		return {'sample': sample, 'ground_truth': ground_truth}
