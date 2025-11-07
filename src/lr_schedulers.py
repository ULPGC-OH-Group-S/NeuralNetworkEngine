import numpy as np


class LRScheduler:
	
	def __init__(self, initial_lr):
		self.initial_lr = initial_lr
		self.lr = initial_lr
		self.epoch = 0
		
	def step(self):
		self.epoch += 1
		return self.lr

class DecayScheduler(LRScheduler):
	
	def __init__(self, initial_lr, decay_rate=0.1):
		super().__init__(initial_lr)
		self.decay_rate = decay_rate
		
	def step(self):
		self.epoch += 1
		self.lr = self.initial_lr / (1 + self.decay_rate * self.epoch)
		return self.lr

class StepScheduler(LRScheduler):
	
	def __init__(self, initial_lr, step_size=10, gamma=0.5):
		super().__init__(initial_lr)
		self.step_size = step_size
		self.gamma = gamma
		
	def step(self):
		self.epoch += 1
		if self.epoch % self.step_size == 0:
			self.lr *= self.gamma
		return self.lr


class CosineAnnealingScheduler(LRScheduler):
	
	def __init__(self, initial_lr, T_max=50):
		super().__init__(initial_lr)
		self.T_max = T_max
		
	def step(self):
		self.epoch += 1
		self.lr = self.initial_lr * (1 + np.cos(np.pi * self.epoch / self.T_max)) / 2
		return self.lr