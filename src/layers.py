import numpy as np


class Layer:
	
	def forward(self, x):
		raise NotImplementedError
	
	def backward(self, grad_output):
		raise NotImplementedError


class FullyConnected(Layer):
	
	def __init__(self, in_features, out_features):
		# Xavier/Glorot initialization
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
	

class Dropout(Layer):
	def __init__(self, p=0.5):
		self.p = p
		self.mask = None
		self.training = True

	def forward(self, x):
		if self.training:
			self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
			return x * self.mask / (1 - self.p)
		else:
			return x

	def backward(self, grad_output):
		if self.training:
			return grad_output * self.mask / (1 - self.p)
		else:
			return grad_output
	

class Sigmoid(Layer):
	
	def __init__(self):
		self.out = None

	def forward(self, x):
		self.out = 1 / (1 + np.exp(-x))
		return self.out

	def backward(self, grad_output):
		return grad_output * self.out * (1 - self.out)
	

class Softmax(Layer):
	def forward(self, input):
		exps = np.exp(input - np.max(input, axis=1, keepdims=True))
		self.out = exps / np.sum(exps, axis=1, keepdims=True)
		return self.out

	def backward(self, grad_output):
		batch_size, num_classes = self.out.shape
		grad_input = np.zeros_like(grad_output)
		for i in range(batch_size):
			s = self.out[i].reshape(-1, 1)
			jacobian = np.diagflat(s) - np.dot(s, s.T)
			grad_input[i] = np.dot(jacobian, grad_output[i])
		return grad_input
	

class ReLU(Layer):
	def __init__(self):
		self.input = None

	def forward(self, x):
		self.input = x
		return np.maximum(0, x)

	def backward(self, grad_output):
		grad_input = grad_output * (self.input > 0)
		return grad_input


class Tanh(Layer):
	def __init__(self):
		self.out = None

	def forward(self, x):
		self.out = np.tanh(x)
		return self.out

	def backward(self, grad_output):
		return grad_output * (1 - self.out ** 2)
