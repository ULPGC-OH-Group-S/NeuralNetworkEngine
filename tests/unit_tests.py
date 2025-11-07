import sys
import os
import numpy as np

from src.layers import FullyConnected
from src.losses import CrossEntropyLoss, MSELoss
from src.network import NeuralNetwork
from src.optimizers import SGD
from src.network import NeuralNetwork


def test_layer_forward_backward():
	# Test FullyConnected layer forward/backward shape
	np.random.seed(0)
	fc = FullyConnected(3, 2)
	x = np.random.randn(5, 3)
	out = fc.forward(x)
	assert out.shape == (5, 2), "Forward shape incorrect"
	grad_out = np.random.randn(5, 2)
	grad_in = fc.backward(grad_out)
	assert grad_in.shape == (5, 3), "Backward shape incorrect"

def test_loss_functions():
	mse = MSELoss()
	y_pred = np.array([[0.5], [0.2]])
	y_true = np.array([[1.0], [0.0]])
	loss = mse.forward(y_pred, y_true)
	assert np.isclose(loss, 0.145), "MSE loss value incorrect"
	ce = CrossEntropyLoss()
	y_pred = np.array([[0.7, 0.2, 0.1]])
	y_true = np.array([[1, 0, 0]])
	loss = ce.forward(y_pred, y_true)
	assert np.isclose(loss, -np.log(0.7)), "CrossEntropy loss value incorrect"

def test_optimizer_step():
	fc = FullyConnected(2, 1)
	old_W = fc.W.copy()
	old_b = fc.b.copy()
	grads = [np.ones_like(fc.W), np.ones_like(fc.b)]
	opt = SGD(lr=0.1)
	opt.update([fc.W, fc.b], grads)
	assert np.allclose(fc.W, old_W - 0.1), "SGD W update incorrect"
	assert np.allclose(fc.b, old_b - 0.1), "SGD b update incorrect"

def test_zero_grad():
	fc = FullyConnected(2, 2)
	fc.dW = np.ones_like(fc.W)
	fc.db = np.ones_like(fc.b)
	net = NeuralNetwork([fc])
	net.zero_grad()
	assert np.all(fc.dW == 0), "zero_grad did not zero dW"
	assert np.all(fc.db == 0), "zero_grad did not zero db"

def test_param_grad_collection():
	fc = FullyConnected(2, 2)
	net = NeuralNetwork([fc])
	params = net.params()
	grads = net.grads()
	assert any(p is fc.W for p in params) and any(p is fc.b for p in params), "params() missing parameters"
	assert any(g is fc.dW for g in grads) and any(g is fc.db for g in grads), "grads() missing gradients"

if __name__ == "__main__":
	test_layer_forward_backward()
	test_loss_functions()
	test_optimizer_step()
	test_zero_grad()
	test_param_grad_collection()

