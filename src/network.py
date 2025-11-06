
class NeuralNetwork:
	def __init__(self, layers):
		self.layers = layers

	def forward(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def backward(self, loss_grad):
		for layer in reversed(self.layers):
			loss_grad = layer.backward(loss_grad)

	def params(self):
		params = []
		for layer in self.layers:
			if hasattr(layer, 'params'):
				params += layer.params()
		return params

	def grads(self):
		grads = []
		for layer in self.layers:
			if hasattr(layer, 'grads'):
				grads += layer.grads()
		return grads

	def zero_grad(self):
		for layer in self.layers:
			if hasattr(layer, 'grads'):
				gs = layer.grads()
				for g in gs:
					if g is not None:
						g[...] = 0
