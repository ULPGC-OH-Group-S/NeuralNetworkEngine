# Neural Network Engine

A simple neural network implementation from scratch using only NumPy.

## Features

- **Layers**: 
  - Dense: FullyConnected, Dropout, BatchNormalization
  - Activation: ReLU, Sigmoid, Tanh, Softmax
  - Convolutional: Conv2D, MaxPooling2D, Flatten
- **Optimizers**: SGD, Adam (with learning rate schedulers)
- **Loss Functions**: MSE, Cross-Entropy
- **Training**: Mini-batch training with early stopping
- **Data Augmentation**: Random rotation, translation, and horizontal flip for images

## Quick Start

### Dense Neural Network

```python
from src.network import NeuralNetwork
from src.layers import FullyConnected, ReLU
from src.optimizers import Adam
from src.losses import MSELoss
from src.trainer import Trainer

# Build network
layers = [
    FullyConnected(13, 64),
    ReLU(),
    FullyConnected(64, 32),
    ReLU(),
    FullyConnected(32, 1)
]
net = NeuralNetwork(layers)

# Train
optimizer = Adam(lr=0.001)
loss_fn = MSELoss()
trainer = Trainer(net, optimizer, loss_fn)
trainer.train(X_train, y_train, X_val, y_val, epochs=100)
```

### Convolutional Neural Network

```python
from src.layers import Conv2D, MaxPooling2D, Flatten, FullyConnected, ReLU, Softmax
from src.losses import CrossEntropyLoss
from src.augmentation import ImageAugmentation

# Build CNN for image classification
layers = [
    Conv2D(in_channels=1, num_filters=16, kernel_size=3),
    ReLU(),
    MaxPooling2D(pool_size=2),
    Conv2D(in_channels=16, num_filters=32, kernel_size=3),
    ReLU(),
    MaxPooling2D(pool_size=2),
    Flatten(),
    FullyConnected(800, 128),
    ReLU(),
    FullyConnected(128, 10),
    Softmax()
]

# Train with data augmentation
augmenter = ImageAugmentation(rotation_range=15, shift_range=0.1, horizontal_flip=True)
trainer.train(X_train, y_train, X_val, y_val, epochs=10, augmenter=augmenter)
```

## Examples

- **IRIS Classification** (`notebooks/iris_experiment.ipynb`) - Classic 3-class classification
- **MNIST Digit Recognition** (`notebooks/mnist_experiment.ipynb`) - Handwritten digit recognition
- **Boston Housing Regression** (`notebooks/boston_regression_experiment.ipynb`) - Regression task
- **Fashion-MNIST CNN** (`notebooks/fashion_mnist_comparison.ipynb`) - DNN vs CNN comparison with data augmentation

## Requirements

```bash
pip install -r requirements.txt
```

## Testing

Comprehensive unit tests covering all layers including CNN components:

```bash
python tests/unit_tests.py
```

**Test Coverage:**
- Dense layers (FullyConnected, Dropout)
- CNN layers (Conv2D, MaxPooling2D, Flatten, BatchNormalization)
- Loss functions and optimizers
- Numerical gradient checking for backpropagation verification
- End-to-end CNN pipeline integration
