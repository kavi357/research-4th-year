import numpy as np
from scipy.spatial.distance import cosine

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 0.0
    L = min(a.size, b.size)
    a2 = a[:L]
    b2 = b[:L]
    return float(1 - cosine(a2, b2))
