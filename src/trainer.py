import numpy as np
from utils import get_batches


class Trainer:

    def __init__(self, network, optimizer, loss_fn):
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=64, patience=15):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
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
            train_loss = np.mean(losses)
            train_losses.append(train_loss)
            val_loss = None
            if X_val is not None and y_val is not None:
                out = self.network.forward(X_val)
                val_loss = self.loss_fn.forward(out, y_val)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")
                # Early stopping
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
