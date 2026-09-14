#!/usr/bin/env python3
"""
Generate Hadamard transform fixtures for the Nim hadamard_rotate_128 suites.

- the production kernel (ext.had_r_128) records the reference values
- the Nim suite (test_q_exl3_hadamard.nim) compares the reimplementation
  against these fixtures
- recording isolates the Hadamard rounding from the GEMM step, a mismatch
  points at the transform alone, in fp32 arithmetic

run as python testgen/gen_exl3_qwen3_00_hadamard.py
"""

from __future__ import annotations

import glob
import os
import sys

import torch
from safetensors.torch import save_file as st_save

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The tests/ tree hosts the quant_utils helpers the generators import.
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))


_venv_python = os.path.dirname(sys.executable)
_venv_bin = os.path.dirname(_venv_python)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + ':' + os.environ.get("PATH", "")
if "CUDA_HOME" not in os.environ:
    sp_base = os.path.join(os.path.dirname(_venv_python), '..', 'lib')
    for d in glob.glob(os.path.join(sp_base, 'python*', 'site-packages', 'nvidia', 'cu*')):
        if os.path.exists(os.path.join(d, 'bin', 'nvcc')):
            os.environ["CUDA_HOME"] = os.path.abspath(d)
            break

from exllamav3.ext import exllamav3_ext as ext  # noqa, the CUDA env setup precedes the import

DEVICE = "cuda:0"
DTYPE = torch.float16
FIXTURE_DIR = os.path.join(_SCRIPT_DIR, "..", "fixtures", "exl3-00-codec", "Qwen3-0.6B-EXL3-5bpw")


def _had_r_128(x, pre_scale=None, post_scale=None, norm=1.0):
    """Returns one ext.had_r_128 output, both scales passed straight through."""
    x = x.contiguous()
    out = torch.empty_like(x)
    ext.had_r_128(x, out, pre_scale, post_scale, norm)
    return out


def save_fixture(name, data):
    """Writes one fixture payload to FIXTURE_DIR as `<name>.safetensor`."""
    path = os.path.join(FIXTURE_DIR, f"{name}.safetensor")
    st_save(data, path)
    print(f"  Saved: {path}")


def main():
    """Records the Hadamard fixture family."""
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # recording cases span several batch sizes and block counts
    cases = [
        ("single_block", 3, 128),
        ("two_blocks", 2, 256),
        ("eight_blocks", 1, 1024),
        ("batch2_eight_blocks", 2, 1024),
        ("odd_blocks", 4, 384),
    ]

    for name, batch, dim in cases:
        print(f"\n=== {name}: batch={batch}, dim={dim} ===")
        x = torch.randn(batch, dim, device=DEVICE, dtype=DTYPE)
        suh = torch.randn(dim, device=DEVICE, dtype=DTYPE)
        svh = torch.randn(dim, device=DEVICE, dtype=DTYPE)

        # the input Hadamard pass carries pre_scale=suh, norm=1.0
        y_pre = _had_r_128(x, pre_scale=suh, norm=1.0)
        # the output Hadamard pass carries post_scale=svh
        y_post = _had_r_128(x, post_scale=svh, norm=1.0)
        # Both scales together
        y_both = _had_r_128(x, pre_scale=suh, post_scale=svh, norm=1.0)
        # Neither scale applied
        y_none = _had_r_128(x, norm=1.0)

        save_fixture(f"hadamard_{name}", {
            "input": x.cpu(),
            "suh": suh.cpu(),
            "svh": svh.cpu(),
            "output_pre": y_pre.cpu(),    # pre_scale only, no post scale
            "output_post": y_post.cpu(),  # post_scale only, no pre scale
            "output_both": y_both.cpu(),  # both scales applied
            "output_none": y_none.cpu()   # no scale applied
        })

    print(f"\nDone. Fixtures in {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
