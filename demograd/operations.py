import numpy as np
from demograd.tensor_engine import Tensor
from demograd.functions import Function

class TensorMax(Function):
    @staticmethod
    def forward(ctx, tensor, axis, keepdims=True):
        ctx.save_for_backward(tensor, axis, keepdims)
        max_val = np.max(tensor.data, axis=axis, keepdims=keepdims)
        return Tensor(max_val)

    @staticmethod
    def backward(ctx, grad_output):
        tensor, axis, keepdims = ctx.saved_tensors
        mask = tensor.data == np.max(tensor.data, axis=axis, keepdims=keepdims)
        return Tensor(mask * grad_output.data)

class TensorSum(Function):
    @staticmethod
    def forward(ctx, tensor, axis, keepdims=True):
        ctx.save_for_backward(tensor.shape, axis, keepdims)
        sum_val = np.sum(tensor.data, axis=axis, keepdims=keepdims)
        return Tensor(sum_val)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, axis, keepdims = ctx.saved_tensors
        output_shape = list(input_shape)
        if not keepdims:
            output_shape[axis] = 1
        grad_expanded = np.broadcast_to(grad_output.data, output_shape)
        return Tensor(grad_expanded)

def tensor_max(tensor, axis, keepdims=True):
    return TensorMax.apply(tensor, axis, keepdims)

def tensor_sum(tensor, axis, keepdims=True):
    return TensorSum.apply(tensor, axis, keepdims)
