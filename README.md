# Neural Network Engine

A simple neural network implementation from scratch using only NumPy.

## Features

- **Layers**: FullyConnected, ReLU, Sigmoid, Tanh, Softmax, Dropout
- **Optimizers**: SGD, Adam (with learning rate schedulers)
- **Loss Functions**: MSE, Cross-Entropy
- **Training**: Mini-batch training with early stopping

## Quick Start

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

## Examples

- **IRIS Classification** (`notebooks/iris_experiment.ipynb`)
- **MNIST Digit Recognition** (`notebooks/mnist_experiment.ipynb`) 
- **Boston Housing Regression** (`notebooks/boston_regression_experiment.ipynb`)

## Requirements

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
python tests/unit_tests.py
```