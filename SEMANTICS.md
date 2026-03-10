# discrete-dists semantics

This document defines the intended behavior of the public distribution API.

## Support

- A support is represented as a half-open interval `[lo, hi)`.
- Querying probabilities outside support returns `0`.
- Sampling returns elements within support.

## `probs(elements)`

- Returns the probability mass for each queried element.
- Returned values are element-wise probabilities, not cumulative probabilities.
- For elements outside support, the result is `0`.
- For defunct distributions, the result is all zeros.

## `sample(rng, n)` and `stratified_sample(rng, n)`

- Return `n` samples from the current distribution.
- Sampling from zero-mass / all-defunct distributions raises `ValueError`.
- Returned values are always inside the live support.

## `sample_without_replacement(rng, n)`

- Returns exactly `n` unique samples or raises `ValueError`.
- It never silently truncates the result.
- Concrete distributions may provide exact implementations when available.

## `is_defunct`

- A distribution is defunct when it cannot produce meaningful samples.
- `Uniform` is defunct when `lo == hi`.
- `Proportional` is defunct when total mass is `0`.
- `MixtureDistribution` is defunct when all children are defunct.

## `isr(target, elements)`

- Computes `target.probs(elements) / self.probs(elements)`.
- If the source probability is positive, the ratio is finite.
- If the source probability is zero and the target probability is positive, the ratio is `inf`.
- If both probabilities are zero, the ratio is `nan`.

## Mixtures

- Mixture probabilities are computed from the active (non-defunct) children only.
- Defunct children are filtered out and the remaining weights are renormalized.
- A mixture with all-defunct children returns zero probabilities and raises on sampling.
