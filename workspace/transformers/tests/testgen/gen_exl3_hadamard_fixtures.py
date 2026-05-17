#!/usr/bin/env python3
"""
Generate Hadamard transform fixtures for testing Nim's hadamard_rotate_128.

Uses the production kernel (ext.had_r_128) as reference. The Nim test
(test_q_exl3_hadamard.nim) compares hadamard_rotate_128 against these fixtures.

This isolates the Hadamard precision issue from the GEMM step, so we can verify
the reimpl matches the kernel exactly in fp32 arithmetic.

Usage:
    CUDA_HOME=... PATH=... python gen_exl3_hadamard_fixtures.py
"""
from __future__ import annotations
import os, sys, json, torch
from safetensors.torch import save_file as st_save

_venv_python = os.path.dirname(sys.executable)
_venv_bin = os.path.dirname(_venv_python)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + ':' + os.environ.get("PATH", "")
if "CUDA_HOME" not in os.environ:
    import glob
    sp_base = os.path.join(os.path.dirname(_venv_python), '..', 'lib')
    for d in glob.glob(os.path.join(sp_base, 'python*', 'site-packages', 'nvidia', 'cu*')):
        if os.path.exists(os.path.join(d, 'bin', 'nvcc')):
            os.environ["CUDA_HOME"] = os.path.abspath(d)
            break

DEVICE = "cuda:0"
DTYPE = torch.float16
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "..", "fixtures", "exl3-hadamard")
os.makedirs(FIXTURE_DIR, exist_ok=True)

from exllamav3.ext import exllamav3_ext as ext

def _had_r_128(x, pre_scale=None, post_scale=None, norm=1.0):
    '''Thin shim calling ext.had_r_128 directly with both scales.'''
    x = x.contiguous()
    out = torch.empty_like(x)
    ext.had_r_128(x, out, pre_scale, post_scale, norm)
    return out

def save_fixture(name, data):
    path = os.path.join(FIXTURE_DIR, f"{name}.safetensor")
    st_save(data, path)
    print(f"  Saved: {path}")

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Test cases: various batch sizes and dimensions
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

    # Input Hadamard: pre_scale=suh, norm=1.0
    y_pre = _had_r_128(x, pre_scale=suh, norm=1.0)
    # Output Hadamard: post_scale=svh
    y_post = _had_r_128(x, post_scale=svh, norm=1.0)
    # Both scales
    y_both = _had_r_128(x, pre_scale=suh, post_scale=svh, norm=1.0)
    # No scale
    y_none = _had_r_128(x, norm=1.0)

    save_fixture(f"hadamard_{name}", {
        "input": x.cpu(),
        "suh": suh.cpu(),
        "svh": svh.cpu(),
        "output_pre": y_pre.cpu(),    # pre_scale only
        "output_post": y_post.cpu(),  # post_scale only
        "output_both": y_both.cpu(),  # both scales
        "output_none": y_none.cpu()  # no scale
    })

print(f"\nDone. Fixtures in {FIXTURE_DIR}")
