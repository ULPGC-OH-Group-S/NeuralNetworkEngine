import numpy as np

class Sigmoid(Layer):
	def __init__(self):
		self.out = None

	def forward(self, x):
		self.out = 1 / (1 + np.exp(-x))
		return self.out

	def backward(self, grad_output):
		return grad_output * self.out * (1 - self.out)

class Layer:
	def forward(self, x):
		raise NotImplementedError
	def backward(self, grad_output):
		raise NotImplementedError

class FullyConnected(Layer):
	def __init__(self, in_features, out_features):
		# Simple Xavier/Glorot initialization
		limit = np.sqrt(6 / (in_features + out_features))
		self.W = np.random.uniform(-limit, limit, (in_features, out_features))
		self.b = np.zeros(out_features)
		self.x = None
		self.dW = None
		self.db = None

	def forward(self, x):
		self.x = x
		return x @ self.W + self.b

	def backward(self, grad_output):
		self.dW = self.x.T @ grad_output / self.x.shape[0]
		self.db = np.mean(grad_output, axis=0)
		grad_input = grad_output @ self.W.T
		return grad_input

	def params(self):
		return [self.W, self.b]

	def grads(self):
		return [self.dW, self.db]

	def forward(self, x):
		self.x = x
		return x @ self.W + self.b

	def backward(self, grad_output):
		# grad_output: (batch, out_features)
		self.dW = self.x.T @ grad_output / self.x.shape[0]
		self.db = np.mean(grad_output, axis=0)
		grad_input = grad_output @ self.W.T
		return grad_input

	def params(self):
		return [self.W, self.b]

	def grads(self):
		return [self.dW, self.db]
	def forward(self, x):
		"""Forward pass through the layer."""
		raise NotImplementedError

	def backward(self, grad_output):
		"""Backward pass through the layer."""
		raise NotImplementedError
