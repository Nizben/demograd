"""
Example: Train a simple neural network using the demograd library.
The network is a two-layer fully connected network for a binary classification task.
"""

import numpy as np
from demograd.tensor_engine import Tensor
from demograd.nn import Linear
from demograd.activations import ReLU, Softmax
from demograd.losses import CrossEntropyLoss, MSELoss
from demograd.optimizers import SGD
from demograd.nn import Sequential

# For matrix multiplication in the Linear layer, we use MatMul from demograd.operations (already imported within the layer).

# Generate synthetic data
def generate_data(N=100, D=2, C=2):
    np.random.seed(0)
    X = np.random.randn(N, D)
    # Create a simple linear boundary and assign labels
    W_true = np.random.randn(D, C)
    scores = np.dot(X, W_true)
    y = np.argmax(scores, axis=1)
    # Wrap labels in a Tensor for compatibility with loss (or leave as numpy array)
    return X, Tensor(y, requires_grad=False, name="labels")

def main():
    # Hyperparameters
    N, D, H, C = 100, 2, 10, 2
    epochs = 100
    lr = 0.01

    # Generate data
    X, labels = generate_data(N, D, C)
    # Wrap input as a Tensor (non-trainable)
    X_tensor = Tensor(X, requires_grad=False, name="X")

    # Build a simple two-layer neural network using Sequential
    model = Sequential(
        Linear(D, H),   # Linear layer from D to H features
        ReLU.apply,     # ReLU activation (we use its apply method as a function)
        Linear(H, C)    # Linear layer from H to C outputs (logits)
    )

    # Gather all parameters from the model
    parameters = model.parameters

    # Define loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(parameters, lr=lr)
    # Optionally, use Adam: from demograd.optimizers import Adam; optimizer = Adam(parameters, lr=lr)

    for epoch in range(epochs):
        # Forward pass
        logits = model(X_tensor)
        loss = loss_fn(logits, labels)
        # Zero gradients
        optimizer.zero_grad()
        # Backward pass (here we assume the loss is scalar so we can call backward with no argument)
        loss.backward()
        # Update parameters
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data}")

if __name__ == "__main__":
    main()
