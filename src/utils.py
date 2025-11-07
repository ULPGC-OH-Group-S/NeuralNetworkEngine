import numpy as np


def one_hot(y, num_classes):
	"""
	Convert integer labels to one-hot encoded vectors.

	Transforms integer class labels into binary vectors where only the
	corresponding class index is set to 1, commonly used for multi-class
	classification with cross-entropy loss.

	Args:
		y (array-like): Integer class labels (0 to num_classes-1)
		num_classes (int): Total number of classes

	Returns:
		numpy.ndarray: One-hot encoded matrix of shape (len(y), num_classes)

	Example:
		>>> one_hot([0, 1, 2], 3)
		array([[1., 0., 0.],
		       [0., 1., 0.],
		       [0., 0., 1.]])
	"""
	y = np.array(y).astype(int)
	return np.eye(num_classes)[y]


def normalize(X):
	"""
	Normalize data to zero mean and unit variance (z-score normalization).

	Applies standard normalization: (X - mean) / std across all features.
	Useful for ensuring all features have similar scales for training stability.

	Args:
		X (array-like): Input data to normalize

	Returns:
		numpy.ndarray: Normalized data with mean ≈ 0 and std ≈ 1
	"""
	X = np.array(X)
	return (X - X.mean()) / X.std()


def standard_scale(data):
	"""
	Standardize data along each feature (column-wise z-score normalization).

	Computes mean and standard deviation for each feature and returns
	normalized data along with scaling parameters for inverse transformation.
	Handles zero standard deviation by setting it to 1.

	Args:
		data (numpy.ndarray): Input data to standardize

	Returns:
		tuple: (scaled_data, mean, std)
			- scaled_data: Standardized data
			- mean: Mean values for each feature
			- std: Standard deviation for each feature
	"""
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	# Prevent division by zero for constant features
	std = np.where(std == 0, 1, std)
	return (data - mean) / std, mean, std


def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None):
	"""
	Split data into training, validation, and test sets.

	Randomly shuffles data and splits into three sets with specified ratios.
	Useful for model training with proper evaluation on unseen data.

	Args:
		X (numpy.ndarray): Input features
		y (numpy.ndarray): Target labels
		train_ratio (float): Fraction of data for training. Default is 0.7.
		val_ratio (float): Fraction of data for validation. Default is 0.15.
		test_ratio (float): Fraction of data for testing. Default is 0.15.
		seed (int, optional): Random seed for reproducible splits

	Returns:
		tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))

	Raises:
		AssertionError: If ratios don't sum to 1.0 (within tolerance)
	"""
	# Validate that ratios sum to 1
	assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
		"Ratios must sum to 1.0"

	n = len(X)
	idx = np.arange(n)

	# Set seed for reproducible results
	if seed is not None:
		np.random.seed(seed)

	# Shuffle indices and apply to data
	np.random.shuffle(idx)
	X, y = X[idx], y[idx]

	# Calculate split indices
	n_train = int(n * train_ratio)
	n_val = int(n * val_ratio)

	# Split data into three sets
	X_train, y_train = X[:n_train], y[:n_train]
	X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
	X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

	return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_batches(X, y, batch_size, shuffle=True, seed=None):
	"""
	Generate mini-batches from input data for training.

	Creates batches of specified size from input data, with optional shuffling
	for each epoch. Essential for mini-batch gradient descent training.

	Args:
		X (numpy.ndarray): Input features
		y (numpy.ndarray): Target labels
		batch_size (int): Size of each batch
		shuffle (bool): Whether to shuffle data before batching. Default is True.
		seed (int, optional): Random seed for reproducible shuffling

	Yields:
		tuple: (X_batch, y_batch) - Mini-batch of features and labels
	"""
	n = len(X)
	idx = np.arange(n)

	# Shuffle indices if requested
	if shuffle:
		if seed is not None:
			np.random.seed(seed)
		np.random.shuffle(idx)

	# Generate batches
	for i in range(0, n, batch_size):
		batch_idx = idx[i:i+batch_size]
		yield X[batch_idx], y[batch_idx]
		
