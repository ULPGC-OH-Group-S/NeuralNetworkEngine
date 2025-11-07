import numpy as np


class SGD:
	
	def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0, lr_scheduler=None):
		self.lr = lr
		self.momentum = momentum
		self.weight_decay = weight_decay
		self.v = {}
		self.lr_scheduler = lr_scheduler

	def update(self, params, grads):
		if self.lr_scheduler:
			self.lr = self.lr_scheduler.step()
		for i, (p, g) in enumerate(zip(params, grads)):
			if self.weight_decay != 0.0:
				g = g + self.weight_decay * p
			if self.momentum != 0.0:
				if i not in self.v:
					self.v[i] = np.zeros_like(g)
				self.v[i] = self.momentum * self.v[i] - self.lr * g
				p += self.v[i]
			else:
				p -= self.lr * g


class Adam:
	
	def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, lr_scheduler=None):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps
		self.weight_decay = weight_decay
		self.m = {}
		self.v = {}
		self.t = 0
		self.lr_scheduler = lr_scheduler

	def update(self, params, grads):
		self.t += 1
		if self.lr_scheduler:
			self.lr = self.lr_scheduler.step()
		for i, (p, g) in enumerate(zip(params, grads)):
			if self.weight_decay != 0.0:
				g = g + self.weight_decay * p
			if i not in self.m:
				self.m[i] = np.zeros_like(g)
				self.v[i] = np.zeros_like(g)
			self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
			self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
			m_hat = self.m[i] / (1 - self.beta1 ** self.t)
			v_hat = self.v[i] / (1 - self.beta2 ** self.t)
			p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)



