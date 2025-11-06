import numpy as np

class MSELoss:
	def forward(self, y_pred, y_true):
		self.y_pred = y_pred
		self.y_true = y_true
		return np.mean((y_pred - y_true) ** 2)

	def backward(self):
		return 2 * (self.y_pred - self.y_true) / self.y_true.size

class CrossEntropyLoss:
	def forward(self, y_pred, y_true):
		# y_pred: (batch, num_classes), y_true: (batch, num_classes) one-hot
		self.y_pred = y_pred
		self.y_true = y_true
		# Add small value to avoid log(0)
		eps = 1e-9
		return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

	def backward(self):
		# Gradient w.r.t. y_pred
		eps = 1e-9
		return -self.y_true / (self.y_pred + eps) / self.y_true.shape[0]
