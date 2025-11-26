#!/usr/bin/env python3
"""
single_core_flops.py — estimate single-core FP32/FP64 GFLOP/s via DGEMM/SGEMM

Usage examples:
  python single_core_flops.py                   # defaults: FP64, sizes 2048,3072,4096
  python single_core_flops.py --dtype f32       # FP32
  python single_core_flops.py --sizes 1024 1536 2048 --repeats 5
  python single_core_flops.py --no-pin          # if CPU affinity pinning causes issues
"""

import os
import time
import argparse
from statistics import median

# --- (1) Parse CLI ---
p = argparse.ArgumentParser()
p.add_argument("--dtype", choices=["f64","f32"], default="f64",
               help="Datatype: f64=FP64 (float64), f32=FP32 (float32).")
p.add_argument("--sizes", type=int, nargs="*", default=[2048, 3072, 4096],
               help="Square matrix sizes to test (pick a few big ones).")
p.add_argument("--repeats", type=int, default=4,
               help="Timing repeats per size; best & median reported.")
p.add_argument("--warmup", type=int, default=1,
               help="Warmup runs per size (not timed).")
p.add_argument("--no-pin", action="store_true",
               help="Disable CPU affinity pinning.", default=True)
p.add_argument("--core", type=int, default=0,
               help="Logical core index to pin to (default 0).")
args = p.parse_args()

# --- (2) Try to force single-threaded BLAS at runtime ---
# NOTE: We set environment variables *and* use threadpoolctl for good measure.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Limit threads via threadpoolctl if available
limit_ctx = None
try:
    from threadpoolctl import threadpool_limits
    limit_ctx = threadpool_limits(limits=1)
except Exception:
    limit_ctx = None  # okay, we'll rely on env vars

# --- (3) Optional: pin this process to a single logical core ---
if not args.no_pin:
    try:
        # Linux / modern Python
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {args.core})
        else:
            # Cross-platform attempt via psutil
            import psutil
            p_self = psutil.Process()
            p_self.cpu_affinity([args.core])
    except Exception as e:
        print(f"[warn] CPU pinning failed ({e}); continuing unpinned.")

# --- (4) Import NumPy after thread controls (safer for some BLAS backends) ---
import numpy as np

dtype = np.float64 if args.dtype == "f64" else np.float32

def gflops_from_dgemm(n: int, t_seconds: float) -> float:
    # DGEMM/SGEMM FLOPs: 2*N^3 (N^3 mults + N^3 adds)
    return (2.0 * (n**3)) / (t_seconds * 1e9)

def bench_size(n: int, repeats: int, warmup: int) -> tuple[float, float]:
    # Allocate once to avoid timing allocation
    A = np.random.random((n, n)).astype(dtype)
    B = np.random.random((n, n)).astype(dtype)
    # Warmup matmuls (not timed)
    for _ in range(warmup):
        C = A @ B
        # prevent optimization removing result
        A[0,0] = C[0,0]
    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        C = A @ B
        t1 = time.perf_counter()
        A[0,0] = C[0,0]  # use the result to keep it "live"
        times.append(t1 - t0)
    best = min(times)
    med = median(times)
    return gflops_from_dgemm(n, best), gflops_from_dgemm(n, med)

def describe_cpu():
    # Small best-effort CPU/affinity summary
    info = []
    try:
        import platform
        info.append(platform.processor())
    except Exception:
        pass
    try:
        if hasattr(os, "sched_getaffinity"):
            aff = sorted(os.sched_getaffinity(0))
            info.append(f"affinity={aff}")
    except Exception:
        pass
    return " | ".join(x for x in info if x)

print("=== Single-core DGEMM/SGEMM GFLOP/s (NumPy BLAS) ===")
print(f"dtype: {args.dtype.upper()}  | sizes: {args.sizes}  | repeats: {args.repeats}  | warmup: {args.warmup}")
cpu_desc = describe_cpu()
if cpu_desc:
    print(f"[info] {cpu_desc}")
if limit_ctx is None:
    print("[info] threadpoolctl not active; relying on env vars for 1 thread.")

results = []
for n in args.sizes:
    try:
        best, med = bench_size(n, args.repeats, args.warmup)
        results.append((n, best, med))
        print(f"N={n:5d}  best: {best:8.2f} GFLOP/s   median: {med:8.2f} GFLOP/s")
    except MemoryError:
        print(f"N={n:5d}  [skipped: not enough RAM]")
    except Exception as e:
        print(f"N={n:5d}  [error: {e}]")

if results:
    # Report the overall best across sizes (often the largest cache-fitting size)
    overall_best = max(results, key=lambda x: x[1])
    print("\n--- Summary ---")
    print("Peak (best across sizes):")
    print(f"  N={overall_best[0]}  best={overall_best[1]:.2f} GFLOP/s  median={overall_best[2]:.2f} GFLOP/s")

print("\nNotes:")
print("• This measures compute-bound GEMM if the size is large enough and BLAS is optimized.")
print("• Ensure CPU is on performance mode (disable deep powersave) for stable results.")
print("• Result approximates single-thread peak GFLOP/s on your BLAS kernel, not absolute ISA peak.")
print("• For apples-to-apples, keep the same dtype and pinning across machines.")
