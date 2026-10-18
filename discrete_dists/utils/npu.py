import numpy as np

def stratified_sample_integers(rng: np.random.Generator, n: int, lo: int, hi: int):
    width = hi - lo
    if width <= 0:
        raise ValueError(f"cannot sample from empty support [{lo}, {hi})")

    u = (np.arange(n) + rng.random(n)) / n
    samples = lo + np.floor(u * width).astype(np.int64)
    return np.minimum(samples, hi - 1)
