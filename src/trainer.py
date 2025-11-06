import numpy as np
from src.utils import get_batches

class Trainer:
    def __init__(self, network, optimizer, loss_fn):
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=64):
        for epoch in range(epochs):
            losses = []
            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                # Forward and backward
                out = self.network.forward(X_batch)
                loss = self.loss_fn.forward(out, y_batch)
                grad = self.loss_fn.backward()
                self.network.backward(grad)
                # Update
                self.optimizer.update(self.network.params(), self.network.grads())
                self.network.zero_grad()
                losses.append(loss)
            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                out = self.network.forward(X_val)
                val_loss = self.loss_fn.forward(out, y_val)
            print(f"Epoch {epoch+1}: train loss={np.mean(losses):.4f}" + (f", val loss={val_loss:.4f}" if val_loss is not None else ""))
