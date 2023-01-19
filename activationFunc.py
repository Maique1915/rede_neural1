import numpy as np

class ActivationFunc:
	"""docstring for ActivationFunc"""
	def __init__(self):
		...

	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def d_sigmoid(self, x):
		return x - x**2

	def i_sigmoid(self, x):
		return np.log(x)-np.log(1-x)

	def relu(self, x):
		return np.maximum(0, x)

	def d_relu(self, x):
		return np.minimum(1, x)

	def tanh(self, x):
		return np.tanh(x)

	def d_tanh(self, x):
		return x - x**2
