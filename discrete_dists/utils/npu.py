import numpy as np

def stratified_sample_integers(rng: np.random.Generator, n: int, lo: int, hi: int):
    buckets = np.linspace(lo, hi, n + 1, endpoint=True, dtype=np.int64)
    samples = [
        rng.integers(buckets[i], buckets[i + 1]) for i in range(n)
    ]
    return np.asarray(samples, dtype=np.int64)
