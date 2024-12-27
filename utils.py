import numpy as np


def topological_sort(tensor):
    """
    Returns a list of functions in topological order of the computation graph
    """
    visited = set()
    sorted_tensors = []
    def topological_sort_helper(tensor):
        if tensor in visited:
            return
        visited.add(tensor)
        for function in tensor.depends_on:
            for inp in function.inputs:
                topological_sort_helper(inp)
            sorted_tensors.append(function)
    
    topological_sort_helper(tensor)
    return sorted_tensors


def broadcast_backward(grad, shape):
    """
    Adjusts the gradient shape to match the original tensor shape by summing over broadcasted dimensions.
    """
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    
    return grad.reshape(shape)