import pickle
import sys

import discrete_dists
import numpy as np
from discrete_dists.proportional import Proportional

print(discrete_dists.__file__, file=sys.stderr)

p = Proportional((10, 18))
p.update(
    np.arange(10, 18),
    np.array([0.5, 1.0, 0.0, 2.0, 3.0, 0.0, 1.5, 4.0]),
)

with open(sys.argv[1], "wb") as f:
    pickle.dump(p, f)
