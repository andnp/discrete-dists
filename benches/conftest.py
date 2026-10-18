import os

# OpenBLAS worker threads spin while benchmarks run and get counted by
# instruction-level profilers, so pin BLAS to one thread before numpy loads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
