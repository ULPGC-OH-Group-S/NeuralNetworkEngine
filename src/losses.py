import numpy as np


class MSELoss:
	"""
	Mean Squared Error (MSE) loss function for regression tasks.

	Computes the average of squared differences between predicted and true values.
	Commonly used for regression problems where the target is continuous.

	Loss formula: L = (1/n) * Σ(y_pred - y_true)²

	Attributes:
		y_pred (numpy.ndarray): Cached predictions from forward pass
		y_true (numpy.ndarray): Cached true labels from forward pass
	"""

	def forward(self, y_pred, y_true):
		"""
		Compute the MSE loss between predictions and true values.

		Args:
			y_pred (numpy.ndarray): Predicted values
			y_true (numpy.ndarray): True target values

		Returns:
			float: Mean squared error loss
		"""
		self.y_pred = y_pred  # Cache for backward pass
		self.y_true = y_true  # Cache for backward pass
		return np.mean((y_pred - y_true) ** 2)

	def backward(self):
		"""
		Compute the gradient of MSE loss with respect to predictions.

		The gradient is: dL/dy_pred = 2 * (y_pred - y_true) / n

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. predictions
		"""
		return 2 * (self.y_pred - self.y_true) / self.y_true.size


class CrossEntropyLoss:
	"""
	Cross-entropy loss function for classification tasks.

	Computes the cross-entropy loss between predicted probabilities and true labels.
	Commonly used for multi-class classification with softmax output layer.
	Includes numerical stability through epsilon addition.

	Loss formula: L = -Σ y_true * log(y_pred + ε)

	Attributes:
		y_pred (numpy.ndarray): Cached predictions from forward pass
		y_true (numpy.ndarray): Cached true labels from forward pass
	"""

	def forward(self, y_pred, y_true):
		"""
		Compute the cross-entropy loss between predictions and true labels.

		Args:
			y_pred (numpy.ndarray): Predicted probabilities of shape (batch_size, num_classes)
			y_true (numpy.ndarray): True labels in one-hot format of same shape

		Returns:
			float: Cross-entropy loss value
		"""
		self.y_pred = y_pred  # Cache for backward pass
		self.y_true = y_true  # Cache for backward pass

		# Add small epsilon to prevent log(0) and ensure numerical stability
		eps = 1e-9
		return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

	def backward(self):
		"""
		Compute the gradient of cross-entropy loss with respect to predictions.

		The gradient is: dL/dy_pred = -y_true / (y_pred + ε) / batch_size

		Returns:
			numpy.ndarray: Gradient of loss w.r.t. predictions
		"""
		eps = 1e-9
		return -self.y_true / (self.y_pred + eps) / self.y_true.shape[0]
