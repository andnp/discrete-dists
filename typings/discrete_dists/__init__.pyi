import numpy as np

class SumTree:
    size: int
    dims: int

    def __new__(cls, *args): ...
    def __init__(self, size: int | None = None, dims: int | None = None): ...
    def update(self, dim: int, idxs: np.ndarray, values: np.ndarray): ...
    def update_single(self, dim: int, idx: int, value: float): ...
    def get_value(self, dim: int, idx: int) -> float: ...
    def get_values(self, dim: int, idxs: np.ndarray) -> np.ndarray: ...
    def dim_total(self, dim: int) -> float: ...
    def all_totals(self) -> np.ndarray: ...
    def total(self, w: np.ndarray) -> float: ...
    def effective_weights(self) -> np.ndarray: ...
    def query(self, v: np.ndarray, w: np.ndarray) -> np.ndarray: ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...
