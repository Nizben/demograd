from demograd.tensor_engine import Tensor
from demograd.functions import Mean

class MSELoss:
    def __call__(self, prediction, target):
        # Ensure target is a Tensor.
        if not isinstance(target, Tensor):
            target = Tensor(target)
        # Compute error and squared error.
        error = prediction - target
        squared_error = error * error
        # Compute the mean of the squared error.
        loss = Mean.apply(squared_error)
        return loss