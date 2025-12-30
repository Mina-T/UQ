import numpy as np

def rescale_list(array, a = 0.0, b = 1.0, eps = 1e-12):
    if isinstance(array, torch.Tensor):
        min_val = array.min()
        max_val = array.max()
        scale = (max_val - min_val).clamp(min=eps)
        return a + (array - min_val) * (b - a) / scale

    if isinstance(array, np.ndarray):
        min_val = np.min(array)
        max_val = np.max(array)
        scale = max(max_val - min_val, eps)
        return a + (array - min_val) * (b - a) / scale

    if isinstance(array, (list, tuple)):
        min_val = min(array)
        max_val = max(array)
        scale = max(max_val - min_val, eps)
        return [
            a + (x - min_val) * (b - a) / scale
            for x in array
        ]

    raise TypeError("Input must be a list, numpy array, or torch tensor.")

