import numpy as np

def one_hot(y, num_classes):
	y = np.array(y).astype(int)
	return np.eye(num_classes)[y]

def normalize(X):
	X = np.array(X)
	return (X - X.mean()) / X.std()

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None):
	assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
	n = len(X)
	idx = np.arange(n)
	if seed is not None:
		np.random.seed(seed)
	np.random.shuffle(idx)
	X, y = X[idx], y[idx]
	n_train = int(n * train_ratio)
	n_val = int(n * val_ratio)
	X_train, y_train = X[:n_train], y[:n_train]
	X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
	X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
	return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def get_batches(X, y, batch_size, shuffle=True, seed=None):
	n = len(X)
	idx = np.arange(n)
	if shuffle:
		if seed is not None:
			np.random.seed(seed)
		np.random.shuffle(idx)
	for i in range(0, n, batch_size):
		batch_idx = idx[i:i+batch_size]
		yield X[batch_idx], y[batch_idx]
