import numpy as np


class SGD:
	"""
	Stochastic Gradient Descent optimizer with momentum and weight decay support.

	Implements the SGD algorithm with optional momentum term for acceleration
	and weight decay for regularization. Supports learning rate scheduling.

	The update rule with momentum is:
	v_t = momentum * v_{t-1} - lr * g_t
	θ_t = θ_{t-1} + v_t

	Attributes:
		lr (float): Learning rate for parameter updates
		momentum (float): Momentum factor for velocity accumulation
		weight_decay (float): L2 regularization coefficient
		v (dict): Velocity vectors for momentum (indexed by parameter)
		lr_scheduler: Optional learning rate scheduler
	"""

	def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0, lr_scheduler=None):
		"""
		Initialize the SGD optimizer.

		Args:
			lr (float): Learning rate. Default is 0.01.
			momentum (float): Momentum factor between 0 and 1. Default is 0.0.
			weight_decay (float): Weight decay (L2 penalty) coefficient. Default is 0.0.
			lr_scheduler: Optional learning rate scheduler object
		"""
		self.lr = lr
		self.momentum = momentum
		self.weight_decay = weight_decay
		self.v = {}  # Velocity vectors for momentum
		self.lr_scheduler = lr_scheduler

	def update(self, params, grads):
		"""
		Update parameters using SGD with momentum and weight decay.

		Applies weight decay to gradients, updates velocities with momentum,
		and performs parameter updates. Supports learning rate scheduling.

		Args:
			params (list): List of parameter arrays to update
			grads (list): List of gradient arrays corresponding to parameters
		"""
		# Update learning rate if scheduler is provided
		if self.lr_scheduler:
			self.lr = self.lr_scheduler.step()

		# Update each parameter
		for i, (p, g) in enumerate(zip(params, grads)):
			# Apply weight decay (L2 regularization)
			if self.weight_decay != 0.0:
				g = g + self.weight_decay * p

			# Apply momentum if specified
			if self.momentum != 0.0:
				# Initialize velocity on first use
				if i not in self.v:
					self.v[i] = np.zeros_like(g)

				# Update velocity and apply to parameters
				self.v[i] = self.momentum * self.v[i] - self.lr * g
				p += self.v[i]
			else:
				# Standard gradient descent without momentum
				p -= self.lr * g


class Adam:
	"""
	Adam (Adaptive Moment Estimation) optimizer.

	Implements the Adam algorithm which computes adaptive learning rates for each
	parameter using estimates of first and second moments of gradients. Combines
	the benefits of AdaGrad and RMSprop, with bias correction for moment estimates.

	The Adam update rules are:
	m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
	v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
	m̂_t = m_t / (1 - β₁^t)
	v̂_t = v_t / (1 - β₂^t)
	θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

	Attributes:
		lr (float): Learning rate
		beta1 (float): Exponential decay rate for first moment estimates
		beta2 (float): Exponential decay rate for second moment estimates
		eps (float): Small constant for numerical stability
		weight_decay (float): L2 regularization coefficient
		m (dict): First moment estimates (indexed by parameter)
		v (dict): Second moment estimates (indexed by parameter)
		t (int): Time step counter for bias correction
		lr_scheduler: Optional learning rate scheduler
	"""

	def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, lr_scheduler=None):
		"""
		Initialize the Adam optimizer.

		Args:
			lr (float): Learning rate. Default is 0.001.
			beta1 (float): First moment decay rate. Default is 0.9.
			beta2 (float): Second moment decay rate. Default is 0.999.
			eps (float): Small constant for numerical stability. Default is 1e-8.
			weight_decay (float): Weight decay (L2 penalty) coefficient. Default is 0.0.
			lr_scheduler: Optional learning rate scheduler object
		"""
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps
		self.weight_decay = weight_decay
		self.m = {}  # First moment estimates
		self.v = {}  # Second moment estimates
		self.t = 0   # Time step counter
		self.lr_scheduler = lr_scheduler

	def update(self, params, grads):
		"""
		Update parameters using the Adam algorithm.

		Computes biased first and second moment estimates, applies bias correction,
		and updates parameters with adaptive learning rates.

		Args:
			params (list): List of parameter arrays to update
			grads (list): List of gradient arrays corresponding to parameters
		"""
		# Increment time step for bias correction
		self.t += 1

		# Update learning rate if scheduler is provided
		if self.lr_scheduler:
			self.lr = self.lr_scheduler.step()

		# Update each parameter
		for i, (p, g) in enumerate(zip(params, grads)):
			# Apply weight decay (L2 regularization)
			if self.weight_decay != 0.0:
				g = g + self.weight_decay * p

			# Initialize moment estimates on first use
			if i not in self.m:
				self.m[i] = np.zeros_like(g)
				self.v[i] = np.zeros_like(g)

			# Update biased first and second moment estimates
			self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
			self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

			# Compute bias-corrected moment estimates
			m_hat = self.m[i] / (1 - self.beta1 ** self.t)
			v_hat = self.v[i] / (1 - self.beta2 ** self.t)

			# Apply Adam update rule
			p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)



