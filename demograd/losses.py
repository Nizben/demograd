import numpy as np
from demograd.tensor_engine import Tensor
from demograd.functions import Log  # ensure you have a differentiable Log
from demograd.operations import tensor_max, tensor_sum

class MSELoss:
    def __call__(self, prediction, target):
        # prediction and target are Tensors
        diff = prediction - target
        loss_val = np.mean(diff.data ** 2)
        # Create a scalar loss tensor
        loss = Tensor(loss_val, requires_grad=True, name="MSELoss")
        return loss


def one_hot(target, num_classes):
    """Converts a 1D array of class indices to one-hot encoded numpy array."""
    N = target.shape[0]
    one_hot_encoded = np.zeros((N, num_classes), dtype=np.float32)
    one_hot_encoded[np.arange(N), target.astype(int)] = 1.0
    return one_hot_encoded

class CrossEntropyLoss:
    def __call__(self, logits, target):
        """
        logits: Tensor of shape (N, C) containing raw scores.
        target: Tensor of shape (N,) containing class indices.
        """
        N, C = logits.data.shape

        # Compute max for numerical stability (differentiable version needed ideally)
        max_logits = tensor_max(logits, axis=1, keepdims=True)  # shape (N, 1)
        shifted_logits = logits - max_logits  # uses overloaded subtraction

        # Compute exponentials using your Tensor exponentiation:
        exp_logits = shifted_logits.__exp__()  # elementwise exp

        # Sum of exponentials over classes:
        sum_exp = tensor_sum(exp_logits, axis=1, keepdims=True)  # shape (N, 1)

        # Compute log(sum_exp) using your differentiable Log function:
        log_sum_exp = Log.apply(sum_exp)  # shape (N, 1)

        # Compute log_softmax: shifted_logits - log_sum_exp (this should be differentiable)
        log_softmax = shifted_logits - log_sum_exp  # shape (N, C)

        # Instead of gathering the correct log-probabilities directly (which would require a differentiable gather),
        # we can convert the target to one-hot encoding:
        one_hot_target = one_hot(target.data, C)  # numpy array of shape (N, C)
        target_tensor = Tensor(one_hot_target, requires_grad=False, name="one_hot_target")

        # Multiply one-hot target with log_softmax elementwise and sum across classes:
        prod = target_tensor * log_softmax  # elementwise multiplication
        # Sum over classes (axis=1) using numpy sum (again, ideally this should be differentiable):
        loss_values = -np.sum(prod.data, axis=1)  # negative log likelihood per sample
        loss_mean = np.mean(loss_values)  # mean over the batch

        loss = Tensor(loss_mean, requires_grad=True, name="CrossEntropyLoss")
        return loss
