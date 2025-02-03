import numpy as np
from demograd.tensor_engine import Tensor

def mse_loss(pred, target):
    diff = pred - target
    # Sum over all elements and average
    loss_val = np.mean(diff.data ** 2)
    return Tensor(loss_val, requires_grad=True)

def cross_entropy_loss(logits, labels):
    # Stabilize logits by subtracting max
    exp_logits = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    batch_size = logits.data.shape[0]
    # Create one-hot encoding of labels
    one_hot = np.zeros_like(softmax)
    one_hot[np.arange(batch_size), labels.data.astype(int)] = 1
    loss_val = -np.sum(one_hot * np.log(softmax + 1e-8)) / batch_size
    return Tensor(loss_val, requires_grad=True)

def accuracy(pred, target):
    """
    Compute accuracy given prediction logits and target labels.
    """
    preds = np.argmax(pred.data, axis=1)
    correct = (preds == target.data).sum()
    return correct / target.data.size
