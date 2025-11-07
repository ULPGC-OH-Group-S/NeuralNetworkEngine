import numpy as np


class Layer:
	"""
	Abstract base class for neural network layers.

	This class defines the interface that all neural network layers must implement.
	It serves as the foundation for building modular, composable network architectures.
	"""

	def forward(self, x):
		"""
		Perform forward propagation through the layer.

		Args:
			x (numpy.ndarray): Input tensor to the layer

		Returns:
			numpy.ndarray: Output tensor after applying the layer's transformation

		Raises:
			NotImplementedError: Must be implemented by concrete layer classes
		"""
		raise NotImplementedError

	def backward(self, grad_output):
		"""
		Perform backward propagation through the layer.

		Computes gradients with respect to layer inputs and parameters (if any).

		Args:
			grad_output (numpy.ndarray): Gradient of the loss with respect to layer output

		Returns:
			numpy.ndarray: Gradient of the loss with respect to layer input

		Raises:
			NotImplementedError: Must be implemented by concrete layer classes
		"""
		raise NotImplementedError


class FullyConnected(Layer):
	"""
	Fully connected (dense) neural network layer.

	Implements a linear transformation: y = xW + b, where W is the weight matrix
	and b is the bias vector. Uses Xavier/Glorot initialization for optimal
	gradient flow during training.

	Attributes:
		W (numpy.ndarray): Weight matrix of shape (in_features, out_features)
		b (numpy.ndarray): Bias vector of shape (out_features,)
		x (numpy.ndarray): Cached input from forward pass for backward computation
		dW (numpy.ndarray): Gradient with respect to weights
		db (numpy.ndarray): Gradient with respect to biases
	"""

	def __init__(self, in_features, out_features):
		"""
		Initialize the fully connected layer with Xavier/Glorot initialization.

		Args:
			in_features (int): Number of input features
			out_features (int): Number of output features
		"""
		# Xavier/Glorot initialization for better gradient flow
		limit = np.sqrt(6 / (in_features + out_features))
		self.W = np.random.uniform(-limit, limit, (in_features, out_features))
		self.b = np.zeros(out_features)

		# Cache for backward pass computation
		self.x = None
		self.dW = None
		self.db = None

	def forward(self, x):
		"""
		Perform forward propagation: y = xW + b.

		Args:
			x (numpy.ndarray): Input tensor of shape (batch_size, in_features)

		Returns:
			numpy.ndarray: Output tensor of shape (batch_size, out_features)
		"""
		self.x = x  # Cache input for backward pass
		return x @ self.W + self.b

	def backward(self, grad_output):
		"""
		Perform backward propagation and compute gradients.

		Computes gradients with respect to weights, biases, and input using
		the chain rule of differentiation.

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. layer output

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. layer input
		"""
		# Compute gradient w.r.t. weights and biases
		self.dW = self.x.T @ grad_output / self.x.shape[0]
		self.db = np.mean(grad_output, axis=0)

		# Compute gradient w.r.t. input for previous layer
		grad_input = grad_output @ self.W.T
		return grad_input

	def params(self):
		"""
		Return list of trainable parameters.

		Returns:
			list: List containing weight matrix and bias vector
		"""
		return [self.W, self.b]

	def grads(self):
		"""
		Return list of computed gradients.

		Returns:
			list: List containing gradients for weights and biases
		"""
		return [self.dW, self.db]
	

class Dropout(Layer):
	"""
	Dropout regularization layer for preventing overfitting.

	During training, randomly sets input elements to zero with probability p,
	and scales the remaining elements by 1/(1-p) to maintain expected output magnitude.
	During inference, passes input unchanged.

	Attributes:
		p (float): Probability of setting an element to zero (dropout rate)
		mask (numpy.ndarray): Binary mask used during forward and backward passes
		training (bool): Whether the layer is in training mode
	"""

	def __init__(self, p=0.5):
		"""
		Initialize the dropout layer.

		Args:
			p (float): Dropout probability between 0 and 1. Default is 0.5.
		"""
		self.p = p
		self.mask = None
		self.training = True

	def forward(self, x):
		"""
		Apply dropout during forward propagation.

		In training mode, randomly zeroes elements and scales remaining ones.
		In evaluation mode, passes input unchanged (inverted dropout).

		Args:
			x (numpy.ndarray): Input tensor

		Returns:
			numpy.ndarray: Output tensor with dropout applied (if training)
		"""
		if self.training:
			# Generate random binary mask and apply inverted dropout
			self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
			return x * self.mask / (1 - self.p)
		else:
			# No dropout during inference
			return x

	def backward(self, grad_output):
		"""
		Apply dropout mask to gradients during backward propagation.

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. layer output

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. layer input with mask applied
		"""
		if self.training:
			return grad_output * self.mask / (1 - self.p)
		else:
			return grad_output
	

class Sigmoid(Layer):
	"""
	Sigmoid activation function layer.

	Applies the sigmoid function element-wise: f(x) = 1 / (1 + exp(-x)).
	Maps input values to the range (0, 1), commonly used for binary classification
	output layers.

	Attributes:
		out (numpy.ndarray): Cached output from forward pass for backward computation
	"""

	def __init__(self):
		"""Initialize the sigmoid activation layer."""
		self.out = None

	def forward(self, x):
		"""
		Apply sigmoid activation function.

		Args:
			x (numpy.ndarray): Input tensor

		Returns:
			numpy.ndarray: Output tensor with sigmoid activation applied
		"""
		self.out = 1 / (1 + np.exp(-x))
		return self.out

	def backward(self, grad_output):
		"""
		Compute gradient of sigmoid activation.

		Uses the identity: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. layer output

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. layer input
		"""
		return grad_output * self.out * (1 - self.out)
	

class Softmax(Layer):
	"""
	Softmax activation function layer.

	Applies the softmax function: f(x_i) = exp(x_i) / sum(exp(x_j)) for all j.
	Converts logits to probability distribution, commonly used for multi-class
	classification output layers. Includes numerical stability via max subtraction.

	Attributes:
		out (numpy.ndarray): Cached output from forward pass for backward computation
	"""

	def forward(self, input):
		"""
		Apply softmax activation function with numerical stability.

		Subtracts the maximum value from inputs before exponentiation to prevent
		numerical overflow while maintaining mathematical correctness.

		Args:
			input (numpy.ndarray): Input tensor of shape (batch_size, num_classes)

		Returns:
			numpy.ndarray: Output probability distribution with same shape as input
		"""
		# Numerical stability: subtract max to prevent overflow
		exps = np.exp(input - np.max(input, axis=1, keepdims=True))
		self.out = exps / np.sum(exps, axis=1, keepdims=True)
		return self.out

	def backward(self, grad_output):
		"""
		Compute gradient of softmax activation using the Jacobian matrix.

		For softmax, the gradient involves computing the full Jacobian matrix
		since each output depends on all inputs.

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. layer output

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. layer input
		"""
		batch_size, num_classes = self.out.shape
		grad_input = np.zeros_like(grad_output)

		# Compute Jacobian for each sample in the batch
		for i in range(batch_size):
			s = self.out[i].reshape(-1, 1)
			# Jacobian matrix: diag(s) - s * s^T
			jacobian = np.diagflat(s) - np.dot(s, s.T)
			grad_input[i] = np.dot(jacobian, grad_output[i])

		return grad_input
	

class ReLU(Layer):
	"""
	Rectified Linear Unit (ReLU) activation function layer.

	Applies the ReLU function element-wise: f(x) = max(0, x).
	Sets negative values to zero while preserving positive values unchanged.
	Most commonly used activation function in deep learning due to its simplicity
	and effectiveness in mitigating vanishing gradient problems.

	Attributes:
		input (numpy.ndarray): Cached input from forward pass for backward computation
	"""

	def __init__(self):
		"""Initialize the ReLU activation layer."""
		self.input = None

	def forward(self, x):
		"""
		Apply ReLU activation function.

		Args:
			x (numpy.ndarray): Input tensor

		Returns:
			numpy.ndarray: Output tensor with ReLU activation applied
		"""
		self.input = x  # Cache input for backward pass
		return np.maximum(0, x)

	def backward(self, grad_output):
		"""
		Compute gradient of ReLU activation.

		Gradient is 1 for positive inputs and 0 for negative inputs.

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. layer output

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. layer input
		"""
		grad_input = grad_output * (self.input > 0)
		return grad_input


class Tanh(Layer):
	"""
	Hyperbolic tangent (tanh) activation function layer.

	Applies the tanh function element-wise: f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)).
	Maps input values to the range (-1, 1), providing zero-centered outputs
	which can be beneficial for certain applications.

	Attributes:
		out (numpy.ndarray): Cached output from forward pass for backward computation
	"""

	def __init__(self):
		"""Initialize the tanh activation layer."""
		self.out = None

	def forward(self, x):
		"""
		Apply tanh activation function.

		Args:
			x (numpy.ndarray): Input tensor

		Returns:
			numpy.ndarray: Output tensor with tanh activation applied
		"""
		self.out = np.tanh(x)
		return self.out

	def backward(self, grad_output):
		"""
		Compute gradient of tanh activation.

		Uses the identity: d/dx tanh(x) = 1 - tanh²(x)

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. layer output

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. layer input
		"""
		return grad_output * (1 - self.out ** 2)
