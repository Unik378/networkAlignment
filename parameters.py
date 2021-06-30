import numpy as np

class Parameters():
	def __init__(self, p=None, size_landmarks=10, max_layer=2, delta = 0.1, base = 2, gamma_s = 1, gamma_a = 1, g1_size = None):
		self.p = p 
		self.size_landmarks = size_landmarks 
		self.max_layer = max_layer 
		self.delta = delta 
		self.base = base 
		self.gamma_s = gamma_s 
		self.gamma_a = gamma_a 
		self.g1_size = g1_size