class NeuralNetwork:
	"""
	A sequential neural network implementation supporting forward and backward propagation.

	This class represents a feedforward neural network composed of sequential layers.
	It provides functionality for forward propagation, backpropagation, parameter management,
	and gradient computation.

	Attributes:
		layers (list): Sequential list of network layers (e.g., FullyConnected, ReLU, etc.)
	"""

	def __init__(self, layers):
		"""
		Initialize the neural network with a list of layers.

		Args:
			layers (list): Sequential list of layer objects that implement forward() and backward() methods
		"""
		self.layers = layers

	def forward(self, x):
		"""
		Perform forward propagation through all network layers.

		Sequentially applies each layer's forward transformation to the input,
		passing the output of one layer as input to the next.

		Args:
			x (numpy.ndarray): Input data with shape (batch_size, input_features)

		Returns:
			numpy.ndarray: Network output after forward propagation through all layers
		"""
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def backward(self, loss_grad):
		"""
		Perform backpropagation through all network layers.

		Propagates gradients backward through the network in reverse layer order,
		computing gradients for each layer's parameters.

		Args:
			loss_grad (numpy.ndarray): Gradient of the loss with respect to network output
		"""
		for layer in reversed(self.layers):
			loss_grad = layer.backward(loss_grad)

	def params(self):
		"""
		Collect all trainable parameters from network layers.

		Iterates through all layers and collects parameters (weights, biases) from
		layers that have trainable parameters.

		Returns:
			list: Flattened list of all parameter arrays (e.g., weights and biases)
		"""
		params = []
		for layer in self.layers:
			if hasattr(layer, 'params'):
				params += layer.params()
		return params

	def grads(self):
		"""
		Collect all gradients from network layers.

		Iterates through all layers and collects computed gradients from
		layers that have trainable parameters.

		Returns:
			list: Flattened list of all gradient arrays corresponding to parameters
		"""
		grads = []
		for layer in self.layers:
			if hasattr(layer, 'grads'):
				grads += layer.grads()
		return grads

	def zero_grad(self):
		"""
		Reset all gradients to zero across network layers.

		This method should be called before each backward pass to clear
		gradients from the previous iteration, preventing gradient accumulation.
		"""
		for layer in self.layers:
			if hasattr(layer, 'grads'):
				gs = layer.grads()
				for g in gs:
					if g is not None:
						g[...] = 0
