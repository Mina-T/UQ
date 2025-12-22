import numpy as np

def rescale_list(values, a, b):
    min_val = min(values)
    max_val = max(values)
    return [a + (x - min_val) / (max_val - min_val) * (b - a) for x in values]


