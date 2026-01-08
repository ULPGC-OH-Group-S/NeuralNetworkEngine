import sys
import os
import numpy as np

from src.layers import FullyConnected, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from src.losses import CrossEntropyLoss, MSELoss
from src.network import NeuralNetwork
from src.optimizers import SGD


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


# ============================================================================
# CNN Layer Tests
# ============================================================================

def test_conv2d_forward_shape():
	"""Test Conv2D layer forward pass output shape."""
	np.random.seed(42)
	# Input: (batch=2, channels=3, height=8, width=8)
	x = np.random.randn(2, 3, 8, 8)
	conv = Conv2D(in_channels=3, num_filters=16, kernel_size=3)
	out = conv.forward(x)
	# Expected output: (batch=2, filters=16, height=6, width=6)
	# out_h = 8 - 3 + 1 = 6
	assert out.shape == (2, 16, 6, 6), f"Conv2D forward shape incorrect: {out.shape}"
	print("✓ Conv2D forward shape test passed")


def test_conv2d_backward_shape():
	"""Test Conv2D layer backward pass gradient shapes."""
	np.random.seed(42)
	x = np.random.randn(2, 3, 8, 8)
	conv = Conv2D(in_channels=3, num_filters=16, kernel_size=3)
	
	out = conv.forward(x)
	grad_output = np.random.randn(*out.shape)
	grad_input = conv.backward(grad_output)
	
	assert grad_input.shape == x.shape, f"Conv2D backward input gradient shape incorrect: {grad_input.shape}"
	assert conv.dW.shape == conv.filters.shape, f"Conv2D weight gradient shape incorrect: {conv.dW.shape}"
	assert conv.db.shape == conv.b.shape, f"Conv2D bias gradient shape incorrect: {conv.db.shape}"
	print("✓ Conv2D backward shape test passed")


def test_conv2d_params():
	"""Test Conv2D layer parameter and gradient collection."""
	conv = Conv2D(in_channels=3, num_filters=8, kernel_size=3)
	params = conv.params()
	assert len(params) == 2, "Conv2D should have 2 parameters (filters and bias)"
	assert params[0].shape == (8, 3, 3, 3), "Conv2D filters shape incorrect"
	assert params[1].shape == (8,), "Conv2D bias shape incorrect"
	
	# Test gradients after backward pass
	x = np.random.randn(2, 3, 10, 10)
	out = conv.forward(x)
	grad_output = np.random.randn(*out.shape)
	conv.backward(grad_output)
	
	grads = conv.grads()
	assert len(grads) == 2, "Conv2D should have 2 gradients"
	assert grads[0].shape == (8, 3, 3, 3), "Conv2D filter gradient shape incorrect"
	assert grads[1].shape == (8,), "Conv2D bias gradient shape incorrect"
	print("✓ Conv2D params test passed")


def test_maxpooling2d_forward_shape():
	"""Test MaxPooling2D layer forward pass output shape."""
	np.random.seed(42)
	x = np.random.randn(2, 16, 8, 8)
	pool = MaxPooling2D(pool_size=2)
	out = pool.forward(x)
	# Expected: (2, 16, 4, 4) since 8/2 = 4
	assert out.shape == (2, 16, 4, 4), f"MaxPooling2D forward shape incorrect: {out.shape}"
	print("✓ MaxPooling2D forward shape test passed")


def test_maxpooling2d_backward_shape():
	"""Test MaxPooling2D layer backward pass gradient shape."""
	np.random.seed(42)
	x = np.random.randn(2, 16, 8, 8)
	pool = MaxPooling2D(pool_size=2)
	
	out = pool.forward(x)
	grad_output = np.random.randn(*out.shape)
	grad_input = pool.backward(grad_output)
	
	assert grad_input.shape == x.shape, f"MaxPooling2D backward shape incorrect: {grad_input.shape}"
	print("✓ MaxPooling2D backward shape test passed")


def test_maxpooling2d_correctness():
	"""Test MaxPooling2D layer computes correct max values."""
	# Create simple input with known values
	x = np.array([[[[1, 2, 3, 4],
	                [5, 6, 7, 8],
	                [9, 10, 11, 12],
	                [13, 14, 15, 16]]]], dtype=np.float32)
	# Shape: (1, 1, 4, 4)
	
	pool = MaxPooling2D(pool_size=2)
	out = pool.forward(x)
	
	# Expected output: max of each 2x2 block
	expected = np.array([[[[6, 8],
	                       [14, 16]]]], dtype=np.float32)
	
	assert np.allclose(out, expected), f"MaxPooling2D values incorrect: {out} vs {expected}"
	print("✓ MaxPooling2D correctness test passed")


def test_flatten_forward():
	"""Test Flatten layer forward pass."""
	x = np.random.randn(4, 16, 5, 5)
	flatten = Flatten()
	out = flatten.forward(x)
	
	# Expected: (4, 16*5*5) = (4, 400)
	assert out.shape == (4, 400), f"Flatten forward shape incorrect: {out.shape}"
	print("✓ Flatten forward test passed")


def test_flatten_backward():
	"""Test Flatten layer backward pass."""
	x = np.random.randn(4, 16, 5, 5)
	flatten = Flatten()
	
	out = flatten.forward(x)
	grad_output = np.random.randn(*out.shape)
	grad_input = flatten.backward(grad_output)
	
	assert grad_input.shape == x.shape, f"Flatten backward shape incorrect: {grad_input.shape}"
	print("✓ Flatten backward test passed")


def test_cnn_pipeline():
	"""Test complete CNN pipeline with Conv2D -> MaxPooling -> Flatten."""
	np.random.seed(42)
	
	# Build mini CNN
	conv = Conv2D(in_channels=1, num_filters=8, kernel_size=3)
	pool = MaxPooling2D(pool_size=2)
	flatten = Flatten()
	fc = FullyConnected(8 * 3 * 3, 10)  # 8 filters, 3x3 after pooling
	
	net = NeuralNetwork([conv, pool, flatten, fc])
	
	# Test forward pass
	x = np.random.randn(4, 1, 8, 8)  # batch=4, channels=1, 8x8 images
	out = net.forward(x)
	
	assert out.shape == (4, 10), f"CNN pipeline output shape incorrect: {out.shape}"
	
	# Test backward pass
	grad_output = np.random.randn(4, 10)
	net.backward(grad_output)
	
	# Check that all parameters have gradients
	params = net.params()
	grads = net.grads()
	assert len(params) > 0, "CNN should have trainable parameters"
	assert len(grads) > 0, "CNN should have gradients"
	
	# Verify gradients are computed (not None and not all zeros)
	for grad in grads:
		assert grad is not None, "Gradient should not be None"
		assert not np.allclose(grad, 0), "Gradient should not be all zeros"
	
	print("✓ CNN pipeline test passed")


def test_batch_normalization_shape():
	"""Test BatchNormalization layer shapes."""
	np.random.seed(42)
	
	# Test with CNN output (4D tensor)
	bn = BatchNormalization(num_features=16)
	x = np.random.randn(8, 16, 7, 7)
	
	out = bn.forward(x)
	assert out.shape == x.shape, f"BatchNorm forward shape incorrect: {out.shape}"
	
	grad_output = np.random.randn(*out.shape)
	grad_input = bn.backward(grad_output)
	assert grad_input.shape == x.shape, f"BatchNorm backward shape incorrect: {grad_input.shape}"
	
	print("✓ BatchNormalization shape test passed")


def test_conv2d_gradient_numerical():
	"""Test Conv2D gradients using numerical gradient checking."""
	np.random.seed(42)
	
	# Very small test case for faster and more accurate computation
	conv = Conv2D(in_channels=1, num_filters=2, kernel_size=2)
	x = np.random.randn(1, 1, 4, 4) * 0.1  # Small values for better numerical stability
	
	# Forward pass
	out = conv.forward(x)
	
	# Backward pass with dummy gradient
	grad_output = np.ones_like(out)  # Simple gradient for easier debugging
	grad_input = conv.backward(grad_output)
	
	# Numerical gradient check for one weight
	epsilon = 1e-4
	filter_idx, channel_idx, i, j = 0, 0, 0, 0
	
	# Store analytical gradient
	analytical_grad = conv.dW[filter_idx, channel_idx, i, j]
	
	# Compute numerical gradient
	original_value = conv.filters[filter_idx, channel_idx, i, j]
	
	conv.filters[filter_idx, channel_idx, i, j] = original_value + epsilon
	out_plus = conv.forward(x)
	loss_plus = np.sum(out_plus * grad_output)
	
	conv.filters[filter_idx, channel_idx, i, j] = original_value - epsilon
	out_minus = conv.forward(x)
	loss_minus = np.sum(out_minus * grad_output)
	
	numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
	
	# Reset weight
	conv.filters[filter_idx, channel_idx, i, j] = original_value
	
	# Check relative error
	rel_error = abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + abs(analytical_grad) + 1e-8)
	
	assert rel_error < 0.05, f"Conv2D gradient check failed: numerical={numerical_grad:.6f}, analytical={analytical_grad:.6f}, rel_error={rel_error:.6f}"
	print(f"✓ Conv2D gradient numerical check passed (rel_error={rel_error:.2e})")


if __name__ == "__main__":
	print("Running basic tests...")
	test_layer_forward_backward()
	test_loss_functions()
	test_optimizer_step()
	test_zero_grad()
	test_param_grad_collection()
	
	print("\nRunning CNN tests...")
	test_conv2d_forward_shape()
	test_conv2d_backward_shape()
	test_conv2d_params()
	test_maxpooling2d_forward_shape()
	test_maxpooling2d_backward_shape()
	test_maxpooling2d_correctness()
	test_flatten_forward()
	test_flatten_backward()
	test_cnn_pipeline()
	test_batch_normalization_shape()
	test_conv2d_gradient_numerical()
	
	print("\n" + "="*50)
	print("All tests passed! ✓")
	print("="*50)
