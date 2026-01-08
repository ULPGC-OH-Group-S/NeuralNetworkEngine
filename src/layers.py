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


class Conv2D(Layer):
	"""
	2D Convolutional layer for processing spatial data (images).

	Applies learnable filters to input feature maps, detecting local patterns.
	Uses valid padding (no padding) and stride of 1 for simplicity.

	Attributes:
		filters (numpy.ndarray): Filter weights of shape (num_filters, in_channels, kernel_size, kernel_size)
		b (numpy.ndarray): Bias vector of shape (num_filters,)
		x (numpy.ndarray): Cached input from forward pass
		dW (numpy.ndarray): Gradient with respect to filters
		db (numpy.ndarray): Gradient with respect to biases
	"""

	def __init__(self, in_channels, num_filters, kernel_size):
		"""
		Initialize the convolutional layer.

		Args:
			in_channels (int): Number of input channels
			num_filters (int): Number of convolutional filters (output channels)
			kernel_size (int): Size of the square convolutional kernel
		"""
		# He initialization for convolutional layers
		limit = np.sqrt(2 / (in_channels * kernel_size * kernel_size))
		self.filters = np.random.randn(num_filters, in_channels, kernel_size, kernel_size) * limit
		self.b = np.zeros(num_filters)
		
		self.x = None
		self.dW = None
		self.db = None

	def forward(self, x):
		"""
		Perform forward convolution.

		Args:
			x (numpy.ndarray): Input tensor of shape (batch_size, in_channels, height, width)

		Returns:
			numpy.ndarray: Output tensor of shape (batch_size, num_filters, out_height, out_width)
		"""
		self.x = x
		batch_size, in_channels, h, w = x.shape
		num_filters, _, kernel_size, _ = self.filters.shape
		
		# Calculate output dimensions (valid padding, stride=1)
		out_h = h - kernel_size + 1
		out_w = w - kernel_size + 1
		
		# Initialize output
		out = np.zeros((batch_size, num_filters, out_h, out_w))
		
		# Perform convolution
		for i in range(out_h):
			for j in range(out_w):
				# Extract patch
				patch = x[:, :, i:i+kernel_size, j:j+kernel_size]
				# Apply all filters to this patch
				for f in range(num_filters):
					out[:, f, i, j] = np.sum(patch * self.filters[f], axis=(1, 2, 3)) + self.b[f]
		
		return out

	def backward(self, grad_output):
		"""
		Perform backward convolution.

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. layer output

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. layer input
		"""
		batch_size, in_channels, h, w = self.x.shape
		num_filters, _, kernel_size, _ = self.filters.shape
		_, _, out_h, out_w = grad_output.shape
		
		# Initialize gradients
		self.dW = np.zeros_like(self.filters)
		self.db = np.zeros_like(self.b)
		grad_input = np.zeros_like(self.x)
		
		# Compute gradients
		for i in range(out_h):
			for j in range(out_w):
				patch = self.x[:, :, i:i+kernel_size, j:j+kernel_size]
				
				for f in range(num_filters):
					# Gradient w.r.t. filters
					self.dW[f] += np.sum(patch * grad_output[:, f:f+1, i:i+1, j:j+1], axis=0)
					# Gradient w.r.t. input
					grad_input[:, :, i:i+kernel_size, j:j+kernel_size] += self.filters[f] * grad_output[:, f:f+1, i:i+1, j:j+1]
		
		# Average gradients over batch
		self.dW /= batch_size
		self.db = np.mean(np.sum(grad_output, axis=(2, 3)), axis=0)
		
		return grad_input

	def params(self):
		"""Return list of trainable parameters."""
		return [self.filters, self.b]

	def grads(self):
		"""Return list of computed gradients."""
		return [self.dW, self.db]


class MaxPooling2D(Layer):
	"""
	2D Max pooling layer for downsampling spatial dimensions.

	Reduces spatial dimensions by taking the maximum value in each pooling window.
	Uses non-overlapping pooling windows (stride = pool_size).

	Attributes:
		pool_size (int): Size of the square pooling window
		mask (numpy.ndarray): Cached indices of max values for backward pass
	"""

	def __init__(self, pool_size=2):
		"""
		Initialize the max pooling layer.

		Args:
			pool_size (int): Size of the pooling window. Default is 2.
		"""
		self.pool_size = pool_size
		self.x = None
		self.mask = None

	def forward(self, x):
		"""
		Perform forward max pooling.

		Args:
			x (numpy.ndarray): Input tensor of shape (batch_size, channels, height, width)

		Returns:
			numpy.ndarray: Output tensor of shape (batch_size, channels, out_height, out_width)
		"""
		self.x = x
		batch_size, channels, h, w = x.shape
		pool_size = self.pool_size
		
		# Calculate output dimensions
		out_h = h // pool_size
		out_w = w // pool_size
		
		# Initialize output and mask
		out = np.zeros((batch_size, channels, out_h, out_w))
		self.mask = np.zeros_like(x)
		
		# Perform max pooling
		for i in range(out_h):
			for j in range(out_w):
				h_start = i * pool_size
				w_start = j * pool_size
				
				# Extract pooling window
				window = x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size]
				
				# Reshape for easier max operation
				window_reshaped = window.reshape(batch_size, channels, -1)
				out[:, :, i, j] = np.max(window_reshaped, axis=2)
				
				# Store mask for backward pass
				max_indices = np.argmax(window_reshaped, axis=2)
				for b in range(batch_size):
					for c in range(channels):
						max_idx = max_indices[b, c]
						h_offset = max_idx // pool_size
						w_offset = max_idx % pool_size
						self.mask[b, c, h_start+h_offset, w_start+w_offset] = 1
		
		return out

	def backward(self, grad_output):
		"""
		Perform backward max pooling.

		Routes gradients only to the positions that had maximum values.

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. layer output

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. layer input
		"""
		batch_size, channels, out_h, out_w = grad_output.shape
		pool_size = self.pool_size
		grad_input = np.zeros_like(self.x)
		
		# Route gradients to max positions
		for i in range(out_h):
			for j in range(out_w):
				h_start = i * pool_size
				w_start = j * pool_size
				
				grad_input[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size] += \
					self.mask[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size] * \
					grad_output[:, :, i:i+1, j:j+1]
		
		return grad_input


class Flatten(Layer):
	"""
	Flatten layer to convert multi-dimensional tensors to 2D.

	Reshapes input from (batch_size, channels, height, width) to (batch_size, features).
	Commonly used between convolutional and fully connected layers.

	Attributes:
		input_shape (tuple): Cached input shape for backward pass
	"""

	def __init__(self):
		"""Initialize the flatten layer."""
		self.input_shape = None

	def forward(self, x):
		"""
		Flatten the input tensor.

		Args:
			x (numpy.ndarray): Input tensor of shape (batch_size, channels, height, width)

		Returns:
			numpy.ndarray: Flattened tensor of shape (batch_size, features)
		"""
		self.input_shape = x.shape
		batch_size = x.shape[0]
		return x.reshape(batch_size, -1)

	def backward(self, grad_output):
		"""
		Reshape gradients back to original input shape.

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. flattened output

		Returns:
			numpy.ndarray: Gradient reshaped to original input dimensions
		"""
		return grad_output.reshape(self.input_shape)


class BatchNormalization(Layer):
	"""
	Batch Normalization layer for normalizing activations.

	Normalizes inputs across the batch dimension, reducing internal covariate shift
	and allowing higher learning rates. Maintains running statistics for inference.

	Attributes:
		gamma (numpy.ndarray): Learnable scale parameter
		beta (numpy.ndarray): Learnable shift parameter
		running_mean (numpy.ndarray): Running average of means for inference
		running_var (numpy.ndarray): Running average of variances for inference
		eps (float): Small constant for numerical stability
		momentum (float): Momentum for running statistics update
		training (bool): Whether layer is in training mode
	"""

	def __init__(self, num_features, eps=1e-5, momentum=0.1):
		"""
		Initialize the batch normalization layer.

		Args:
			num_features (int): Number of features/channels to normalize
			eps (float): Small constant for numerical stability. Default is 1e-5.
			momentum (float): Momentum for running statistics. Default is 0.1.
		"""
		self.gamma = np.ones(num_features)
		self.beta = np.zeros(num_features)
		self.running_mean = np.zeros(num_features)
		self.running_var = np.ones(num_features)
		self.eps = eps
		self.momentum = momentum
		self.training = True
		
		# Cache for backward pass
		self.x_norm = None
		self.x_centered = None
		self.std = None
		self.dgamma = None
		self.dbeta = None

	def forward(self, x):
		"""
		Perform forward batch normalization.

		During training: normalize using batch statistics and update running stats.
		During inference: normalize using running statistics.

		Args:
			x (numpy.ndarray): Input tensor of shape (batch_size, num_features) or
			                   (batch_size, channels, height, width)

		Returns:
			numpy.ndarray: Normalized and scaled output with same shape as input
		"""
		# Handle 4D input (for CNNs)
		if x.ndim == 4:
			batch_size, channels, height, width = x.shape
			# Reshape to (batch_size * height * width, channels)
			x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, channels)
			out_reshaped = self._forward_2d(x_reshaped)
			# Reshape back to (batch_size, channels, height, width)
			return out_reshaped.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)
		else:
			return self._forward_2d(x)

	def _forward_2d(self, x):
		"""Helper function for 2D forward pass."""
		if self.training:
			# Compute batch statistics
			mean = np.mean(x, axis=0)
			var = np.var(x, axis=0)
			
			# Update running statistics
			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
			
			# Normalize
			self.x_centered = x - mean
			self.std = np.sqrt(var + self.eps)
			self.x_norm = self.x_centered / self.std
		else:
			# Use running statistics for inference
			self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
		
		# Scale and shift
		out = self.gamma * self.x_norm + self.beta
		return out

	def backward(self, grad_output):
		"""
		Perform backward batch normalization.

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. layer output

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. layer input
		"""
		# Handle 4D input (for CNNs)
		if grad_output.ndim == 4:
			batch_size, channels, height, width = grad_output.shape
			grad_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, channels)
			grad_input_reshaped = self._backward_2d(grad_reshaped)
			return grad_input_reshaped.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)
		else:
			return self._backward_2d(grad_output)

	def _backward_2d(self, grad_output):
		"""Helper function for 2D backward pass."""
		batch_size = grad_output.shape[0]
		
		# Gradient w.r.t. gamma and beta
		self.dgamma = np.sum(grad_output * self.x_norm, axis=0)
		self.dbeta = np.sum(grad_output, axis=0)
		
		# Gradient w.r.t. normalized x
		dx_norm = grad_output * self.gamma
		
		# Gradient w.r.t. variance
		dvar = np.sum(dx_norm * self.x_centered, axis=0) * -0.5 * (self.std ** -3)
		
		# Gradient w.r.t. mean
		dmean = np.sum(dx_norm * -1 / self.std, axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)
		
		# Gradient w.r.t. input
		dx = dx_norm / self.std + dvar * 2 * self.x_centered / batch_size + dmean / batch_size
		
		return dx

	def params(self):
		"""Return list of trainable parameters."""
		return [self.gamma, self.beta]

	def grads(self):
		"""Return list of computed gradients."""
		return [self.dgamma, self.dbeta]


class AveragePooling2D(Layer):
	"""
	2D Average pooling layer for downsampling spatial dimensions.

	Reduces spatial dimensions by taking the average value in each pooling window.
	Uses non-overlapping pooling windows (stride = pool_size).

	Attributes:
		pool_size (int): Size of the square pooling window
		x (numpy.ndarray): Cached input from forward pass
	"""

	def __init__(self, pool_size=2):
		"""
		Initialize the average pooling layer.

		Args:
			pool_size (int): Size of the pooling window. Default is 2.
		"""
		self.pool_size = pool_size
		self.x = None

	def forward(self, x):
		"""
		Perform forward average pooling.

		Args:
			x (numpy.ndarray): Input tensor of shape (batch_size, channels, height, width)

		Returns:
			numpy.ndarray: Output tensor of shape (batch_size, channels, out_height, out_width)
		"""
		self.x = x
		batch_size, channels, h, w = x.shape
		pool_size = self.pool_size
		
		# Calculate output dimensions
		out_h = h // pool_size
		out_w = w // pool_size
		
		# Initialize output
		out = np.zeros((batch_size, channels, out_h, out_w))
		
		# Perform average pooling
		for i in range(out_h):
			for j in range(out_w):
				h_start = i * pool_size
				w_start = j * pool_size
				
				# Extract pooling window and compute average
				window = x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size]
				out[:, :, i, j] = np.mean(window, axis=(2, 3))
		
		return out

	def backward(self, grad_output):
		"""
		Perform backward average pooling.

		Distributes gradients evenly across all positions in each pooling window.

		Args:
			grad_output (numpy.ndarray): Gradient of loss w.r.t. layer output

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. layer input
		"""
		batch_size, channels, out_h, out_w = grad_output.shape
		pool_size = self.pool_size
		grad_input = np.zeros_like(self.x)
		
		# Distribute gradients evenly to all positions in pooling window
		for i in range(out_h):
			for j in range(out_w):
				h_start = i * pool_size
				w_start = j * pool_size
				
				# Distribute gradient evenly across the pooling window
				grad_input[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size] += \
					grad_output[:, :, i:i+1, j:j+1] / (pool_size * pool_size)
		
		return grad_input
