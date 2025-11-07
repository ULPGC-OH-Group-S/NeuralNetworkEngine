import numpy as np
from utils import get_batches


class Trainer:
    """
    Training orchestrator for neural networks with support for validation and early stopping.

    This class handles the complete training loop including forward propagation,
    loss computation, backpropagation, parameter updates, and monitoring.
    Supports early stopping based on validation loss to prevent overfitting.

    Attributes:
        network: Neural network model to train
        optimizer: Optimization algorithm for parameter updates
        loss_fn: Loss function for computing training objective
    """

    def __init__(self, network, optimizer, loss_fn):
        """
        Initialize the trainer with network, optimizer, and loss function.

        Args:
            network: Neural network instance with forward/backward methods
            optimizer: Optimizer instance (e.g., SGD, Adam) for parameter updates
            loss_fn: Loss function instance for computing training objective
        """
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=64, patience=15):
        """
        Execute the complete training loop with optional validation and early stopping.

        Performs mini-batch gradient descent with configurable batch size, tracks
        training progress, and implements early stopping based on validation loss
        to prevent overfitting.

        Args:
            X_train (numpy.ndarray): Training input data
            y_train (numpy.ndarray): Training target labels
            X_val (numpy.ndarray, optional): Validation input data for monitoring
            y_val (numpy.ndarray, optional): Validation target labels
            epochs (int): Maximum number of training epochs. Default is 10.
            batch_size (int): Size of mini-batches for gradient computation. Default is 64.
            patience (int): Early stopping patience (epochs without improvement). Default is 15.

        Returns:
            tuple: (train_losses, val_losses) - Lists of loss values per epoch
        """
        # Initialize early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        # Main training loop
        for epoch in range(epochs):
            losses = []

            # Mini-batch training
            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                # Forward pass: compute predictions and loss
                out = self.network.forward(X_batch)
                loss = self.loss_fn.forward(out, y_batch)

                # Backward pass: compute gradients
                grad = self.loss_fn.backward()
                self.network.backward(grad)

                # Parameter update and gradient reset
                self.optimizer.update(self.network.params(), self.network.grads())
                self.network.zero_grad()

                losses.append(loss)

            # Compute average training loss for this epoch
            train_loss = np.mean(losses)
            train_losses.append(train_loss)

            # Validation evaluation and early stopping
            val_loss = None
            if X_val is not None and y_val is not None:
                # Compute validation loss (no gradient computation needed)
                out = self.network.forward(X_val)
                val_loss = self.loss_fn.forward(out, y_val)
                val_losses.append(val_loss)

                print(f"Epoch {epoch+1}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}: train loss={train_loss:.4f}")

        return train_losses, val_losses
