"""Standalone DGEMM driver for profiling: run under `nsys profile python bench_dgemm.py`.

Runs warmup iterations, then timed iterations, through the GEMMul8 INT8 shim.
Note: each shim call also mallocs and copies A/B/C host<->device, which shows up in
the timeline; use the kernel summary (cuda_gpu_kern_sum) to isolate GEMMul8 kernels.
"""

import argparse
import ctypes

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--lib", required=True)
p.add_argument("--n", type=int, default=4096)
p.add_argument("--iters", type=int, default=5)
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--num-moduli", type=int, default=15)
p.add_argument("--accurate", action="store_true", help="use accurate mode (default: fast mode)")
args = p.parse_args()

lib = ctypes.CDLL(args.lib)
dp = ctypes.POINTER(ctypes.c_double)
rng = np.random.default_rng(0)
a = np.asfortranarray(rng.standard_normal((args.n, args.n)))
b = np.asfortranarray(rng.standard_normal((args.n, args.n)))
c = np.zeros((args.n, args.n), order="F")

for i in range(args.warmup + args.iters):
    rc = lib.gemmul8_dgemm_int8(
        args.n, args.n, args.n, a.ctypes.data_as(dp), b.ctypes.data_as(dp), c.ctypes.data_as(dp),
        args.num_moduli, int(not args.accurate),
    )
    if rc != 0:
        raise SystemExit(f"iteration {i}: gemmul8_dgemm_int8 failed with {rc}")
print(f"done: n={args.n} iters={args.iters} warmup={args.warmup}")
