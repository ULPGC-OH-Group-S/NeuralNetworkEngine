import numpy as np


class LRScheduler:
	"""
	Base class for learning rate scheduling algorithms.

	Provides the common interface and functionality for all learning rate schedulers.
	Tracks the current epoch and maintains both initial and current learning rates.

	Attributes:
		initial_lr (float): The initial learning rate value
		lr (float): The current learning rate value
		epoch (int): Current epoch/step counter
	"""

	def __init__(self, initial_lr):
		"""
		Initialize the learning rate scheduler.

		Args:
			initial_lr (float): Initial learning rate value
		"""
		self.initial_lr = initial_lr
		self.lr = initial_lr
		self.epoch = 0

	def step(self):
		"""
		Update the learning rate and increment epoch counter.

		This base implementation maintains constant learning rate.
		Subclasses should override this method to implement specific scheduling logic.

		Returns:
			float: Updated learning rate value
		"""
		self.epoch += 1
		return self.lr


class DecayScheduler(LRScheduler):
	"""
	Exponential decay learning rate scheduler.

	Gradually reduces learning rate using the formula:
	lr = initial_lr / (1 + decay_rate * epoch)

	This provides smooth exponential decay that asymptotically approaches zero.

	Attributes:
		decay_rate (float): Controls the rate of decay
	"""

	def __init__(self, initial_lr, decay_rate=0.1):
		"""
		Initialize the decay scheduler.

		Args:
			initial_lr (float): Initial learning rate value
			decay_rate (float): Decay rate coefficient. Default is 0.1.
		"""
		super().__init__(initial_lr)
		self.decay_rate = decay_rate

	def step(self):
		"""
		Update learning rate using exponential decay formula.

		Returns:
			float: Updated learning rate value
		"""
		self.epoch += 1
		self.lr = self.initial_lr / (1 + self.decay_rate * self.epoch)
		return self.lr


class StepScheduler(LRScheduler):
	"""
	Step-wise learning rate scheduler.

	Reduces learning rate by a fixed factor (gamma) every step_size epochs.
	Provides piecewise constant learning rate schedule commonly used in training.

	Attributes:
		step_size (int): Number of epochs between rate reductions
		gamma (float): Multiplicative factor for learning rate reduction
	"""

	def __init__(self, initial_lr, step_size=10, gamma=0.5):
		"""
		Initialize the step scheduler.

		Args:
			initial_lr (float): Initial learning rate value
			step_size (int): Interval for learning rate reduction. Default is 10.
			gamma (float): Multiplicative decay factor. Default is 0.5.
		"""
		super().__init__(initial_lr)
		self.step_size = step_size
		self.gamma = gamma

	def step(self):
		"""
		Update learning rate with step-wise reduction.

		Multiplies current learning rate by gamma every step_size epochs.

		Returns:
			float: Updated learning rate value
		"""
		self.epoch += 1
		if self.epoch % self.step_size == 0:
			self.lr *= self.gamma
		return self.lr


class CosineAnnealingScheduler(LRScheduler):
	"""
	Cosine annealing learning rate scheduler.

	Adjusts learning rate following a cosine function, providing smooth
	transitions between high and low rates. Commonly used for achieving
	better convergence in later training stages.

	Formula: lr = initial_lr * (1 + cos(π * epoch / T_max)) / 2

	Attributes:
		T_max (int): Maximum number of epochs for one cosine cycle
	"""

	def __init__(self, initial_lr, T_max=50):
		"""
		Initialize the cosine annealing scheduler.

		Args:
			initial_lr (float): Initial learning rate value
			T_max (int): Period of the cosine cycle. Default is 50.
		"""
		super().__init__(initial_lr)
		self.T_max = T_max

	def step(self):
		"""
		Update learning rate using cosine annealing formula.

		Returns:
			float: Updated learning rate value
		"""
		self.epoch += 1
		self.lr = self.initial_lr * (1 + np.cos(np.pi * self.epoch / self.T_max)) / 2
		return self.lr