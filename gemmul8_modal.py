"""Run GEMMul8 serverlessly on Modal.

This tooling lives on its own branch (modal-tooling) and is pointed at any GEMMul8
checkout via GEMMUL8_SRC, so library branches never contain it:

    export GEMMUL8_SRC=../GEMMul8                 # or an agent's worktree
    modal run gemmul8_modal.py::main              # build image, run samples + a DGEMM check
    modal run gemmul8_modal.py::profile           # Nsight Systems trace -> Volume gemmul8-profiles
    modal deploy gemmul8_modal.py                 # expose `dgemm` (bakes in GEMMUL8_SRC as of deploy)

The library is compiled for ONE GPU arch (Makefile uses a single -gencode), so
GPU and GPU_ARCH below must match.
"""

import os
from pathlib import Path

import modal

# Use "B200", not "B200+": B200+ may land on a B300 (sm_103, needs CUDA 13.1+).
GPU = "B200"  # sm_100 (Blackwell): INT8 + FP8 both supported. L4 = "89", H100 = "90" (no FP8 in GEMMul8).
GPU_ARCH = "100"
OPS = "gemm"  # keep the build short; use "all" for every routine
OZ2_BACKENDS = "INT8 FP8"

TOOLS_DIR = Path(__file__).resolve().parent
if modal.is_local():
    if "GEMMUL8_SRC" not in os.environ:
        raise SystemExit("Set GEMMUL8_SRC to the GEMMul8 checkout to build, e.g. ../GEMMul8")
    SRC_DIR = Path(os.environ["GEMMUL8_SRC"]).expanduser().resolve()
    if not (SRC_DIR / "include" / "gemmul8.hpp").is_file():
        raise SystemExit(f"GEMMUL8_SRC={SRC_DIR} does not look like a GEMMul8 checkout")
else:
    SRC_DIR = Path("/unused")  # image is already built when this module loads in a container

SRC = "/gemmul8"  # library source inside the container
TOOLS = "/tools"  # this tooling inside the container

# CUDA >= 12.9 is required by GEMMul8; Modal's host driver supports CUDA 13.x, so 12.9 is fine.
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("make", "cuda-nsight-systems-12-9")  # nsys CLI from the image's CUDA apt repo
    .run_commands(
        "ln -sf \"$(find /opt/nvidia/nsight-systems -path '*/bin/nsys' | head -n 1)\" /usr/local/bin/nsys",
        "nsys --version",
    )
    .pip_install("numpy")
    .add_local_dir(SRC_DIR, SRC, copy=True, ignore=[".git", "build", "lib", "**/__pycache__"])
    # No GPU exists at build time, so pass BACKEND/GPU_ARCH explicitly instead of
    # relying on nvidia-smi auto-detection. libcuda/libnvidia-ml link via the toolkit stubs.
    .run_commands(
        f"cd {SRC} && make -j$(nproc) BACKEND=cuda GPU_ARCH={GPU_ARCH} "
        f"OPS='{OPS}' OZ2_BACKENDS='{OZ2_BACKENDS}'",
        f"cd {SRC}/sample && make BACKEND=cuda GPU_ARCH={GPU_ARCH}",
    )
    .add_local_file(TOOLS_DIR / "dgemm_shim.cu", f"{TOOLS}/dgemm_shim.cu", copy=True)
    .run_commands(
        f"cd {SRC} && nvcc -std=c++20 -O3 -shared -Xcompiler -fPIC "
        f"-gencode arch=compute_{GPU_ARCH},code=sm_{GPU_ARCH} -Iinclude "
        f"{TOOLS}/dgemm_shim.cu -o lib/libdgemm_shim.so "
        f"lib/libgemmul8.a -lcublas -lcublasLt -lcudart -lcuda -lnvidia-ml -ldl",
    )
    # Not copied into the image: mounted at container start, so editing it needs no rebuild.
    .add_local_file(TOOLS_DIR / "bench_dgemm.py", f"{TOOLS}/bench_dgemm.py")
)

app = modal.App("gemmul8", image=image)

# Nsight Systems reports persist here; browse with `modal volume ls gemmul8-profiles`.
profiles = modal.Volume.from_name("gemmul8-profiles", create_if_missing=True)
PROFILE_DIR = "/profiles"


@app.function(gpu=GPU, timeout=600)
def run_samples() -> str:
    import subprocess

    out = []
    for exe in ("dgemm_cuBLAS_int8", "dgemm_cuBLASLt_fp8"):
        r = subprocess.run([f"{SRC}/sample/{exe}"], capture_output=True, text=True)
        out.append(f"--- {exe} (exit {r.returncode}) ---\n{r.stdout[-600:]}{r.stderr[-600:]}")
    return "\n".join(out)


@app.cls(gpu=GPU, timeout=600, scaledown_window=60)
class GEMMul8:
    @modal.enter()
    def load(self):
        import ctypes

        self.lib = ctypes.CDLL(f"{SRC}/lib/libdgemm_shim.so")

    @modal.method()
    def dgemm(self, a, b, num_moduli: int = 15, fastmode: bool = True):
        """C = A @ B in emulated FP64 on INT8 tensor cores."""
        import ctypes

        import numpy as np

        a = np.asfortranarray(a, dtype=np.float64)
        b = np.asfortranarray(b, dtype=np.float64)
        m, k = a.shape
        k2, n = b.shape
        assert k == k2, "inner dimensions differ"
        c = np.zeros((m, n), dtype=np.float64, order="F")
        p = ctypes.POINTER(ctypes.c_double)
        rc = self.lib.gemmul8_dgemm_int8(
            m, n, k, a.ctypes.data_as(p), b.ctypes.data_as(p), c.ctypes.data_as(p),
            num_moduli, int(fastmode),
        )
        if rc != 0:
            raise RuntimeError(f"gemmul8_dgemm_int8 failed: {rc}")
        return c


@app.function(gpu=GPU, timeout=1800, volumes={PROFILE_DIR: profiles})
def nsys_profile(label: str, bench_args: list[str]) -> dict:
    """Run bench_dgemm.py under Nsight Systems and save the report on the Volume.

    `label` names the run's parameters; a UTC timestamp plus a random suffix is appended
    here (on the remote side) so runs with identical parameters never collide, whoever
    calls this. Returns the report's path on the Volume and the kernel/memcpy summary.
    """
    import secrets
    import subprocess
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{label}_{stamp}_{secrets.token_hex(3)}"
    out = f"{PROFILE_DIR}/{name}"
    subprocess.run(
        [
            "nsys", "profile",
            "--trace=cuda,nvtx,cublas,osrt",
            # CPU sampling / context-switch tracing need perf_event access that a
            # container may not grant; CUDA tracing via CUPTI does not.
            "--sample=none", "--cpuctxsw=none",
            "-o", out,  # no --force-overwrite: a name collision should fail, not clobber
            "python", f"{TOOLS}/bench_dgemm.py", "--lib", f"{SRC}/lib/libdgemm_shim.so",
            *bench_args,
        ],
        check=True,
    )
    stats = subprocess.run(
        ["nsys", "stats", "--report", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum", f"{out}.nsys-rep"],
        capture_output=True, text=True, check=True,
    ).stdout
    profiles.commit()
    return {"report": f"{name}.nsys-rep", "stats": stats}


def src_revision() -> str:
    """Short commit of GEMMUL8_SRC, with -dirty if it has uncommitted changes."""
    import subprocess

    def git(*args):
        return subprocess.run(["git", "-C", str(SRC_DIR), *args],
                              capture_output=True, text=True).stdout.strip()

    rev = git("rev-parse", "--short", "HEAD") or "nogit"
    return f"{rev}-dirty" if git("status", "--porcelain") else rev


@app.local_entrypoint()
def profile(n: int = 4096, iters: int = 5, num_moduli: int = 15, accurate: bool = False,
            label: str = ""):
    """modal run gemmul8_modal.py::profile --n 8192"""
    label = label or f"dgemm_int8_n{n}_mod{num_moduli}_{'accu' if accurate else 'fast'}_{GPU}"
    label = f"{label}_{src_revision()}"
    bench_args = ["--n", str(n), "--iters", str(iters), "--num-moduli", str(num_moduli)]
    if accurate:
        bench_args.append("--accurate")
    result = nsys_profile.remote(label, bench_args)
    print(result["stats"])
    print(f"report on Volume gemmul8-profiles: {result['report']}\n"
          f"fetch with: modal volume get gemmul8-profiles {result['report']}")


@app.local_entrypoint()
def main(n: int = 1024):
    import numpy as np

    print(run_samples.remote())

    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    c = GEMMul8().dgemm.remote(a, b)
    ref = a @ b
    print(f"n={n}  max relative error vs numpy FP64: "
          f"{np.max(np.abs(c - ref)) / np.max(np.abs(ref)):.3e}")
