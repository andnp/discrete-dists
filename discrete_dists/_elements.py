import numpy as np
import numpy.typing as npt

_INT64 = np.dtype(np.int64)


def as_elements(elements: npt.ArrayLike) -> np.ndarray:
    """
    Convert element indices to an int64 array, rejecting anything that is not
    integer-valued. Empty input is accepted for any dtype, since `np.asarray([])`
    is float64.
    """
    if isinstance(elements, np.ndarray) and elements.dtype is _INT64:
        return elements

    a = np.asarray(elements)
    if a.size == 0 or a.dtype.kind in "iu":
        return a.astype(np.int64, copy=False)

    raise TypeError(f"elements must be integers, got dtype {a.dtype}")
