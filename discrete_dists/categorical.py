from discrete_dists.proportional import Proportional


class Categorical(Proportional):
    """
    Convenience alias for `Proportional` using standard probability terminology.

    A categorical distribution samples discrete elements proportional to their
    assigned nonnegative weights over a finite support.
    """
