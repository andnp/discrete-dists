# discrete dists

Simple utilities for defining complex distributions over discrete elements.
Backed by a fast sum-tree implementation written in Rust.


## Getting Started

```bash
pip install discrete-dists
```

## API
### Uniform Distribution
A very simple wrapper over `np.random.default_rng().integers`, conforming to the `Distribution` API defined in this library.
This wrapper additionally introduces importance sampling ratio calculations, sampling without replacement, and stratified sampling.

```python
import numpy as np
from discrete_dists.uniform import Uniform

rng = np.random.default_rng(0)

u = Uniform(100)

# sampling
print(u.sample(rng, 10))
print(u.stratified_sample(rng, 10))
print(u.sample_without_replacement(rng, 10))

# importance sampling ratio
other = Uniform(10)

items = [0, 3, 8]
isrs = u.isr(other, items)

# updating the support
u.update_single(150)
```

### Proportional Distribution
Sample proportional to a list of values.

```python
from discrete_dists.proportional import Proportional

p = Proportional(5)

# set the values to sample proportional to
p.update(idxs=[0, 2], values=[1, 2])
# approximately 33% of values are 0, and 66% are 2
print(p.sample(rng, 10000))

p.update(idxs=[1], values=[2])
# approximately 20% are 0, 40% are 1, and 40% are 2
print(p.sample(rng, 10000))
```

### Mixture Distribution
Mix together arbitrary distributions with arbitrary supports.

```python
from discrete_dists.proportional import Proportional
from discrete_dists.uniform import Uniform
from discrete_dists.mixture import MixtureDistribution, SubDistribution

prop = Proportional(100)
m = MixtureDistribution([
    SubDistribution(d=prop, p=0.2),
    SubDistribution(d=Uniform(10), p=0.8),
])

prop.update(idxs=np.arange(100), values=100-np.arange(100))

print(m.sample(rng, 10000))
```
