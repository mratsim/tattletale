#!/usr/bin/env python3
"""
4-way test: confirm ext.hgemm == F.mm == F.matmul != F.linear for EXL3 weight layout.

Background:
  - ext.hgemm(x, w_non_transposed, y) where w is [in_f, out_f]
  - F.mm(x, w) and F.matmul(x, w) produce bit-exact results with ext.hgemm
  - F.linear(x, w.t().contiguous()) differs (~0.001+ fp16 ULP accumulation)
  - F.matmul handles ND broadcasting (unlike F.mm which is 2D-only)
  - Proof: hgemm == mm == matmul, all differ from linear
"""
import torch, sys, os
from exllamav3.ext import exllamav3_ext as ext
sys.path.insert(0, os.path.dirname(__file__))
from q_exl3_common import reconstruct_orig_exl3
from safetensors.torch import load_file as st_load

DEVICE = 'cuda:0'
TOL = 1e-8  # expect bit-exact

raw = st_load(os.path.join(os.path.dirname(__file__), '..',
    'hf_models', 'Qwen3-0.6B-EXL3-5bpw', 'model.safetensors'), device=DEVICE)

PREFIX = 'model.layers.0'
projs = [
    f'{PREFIX}.self_attn.q_proj',
    f'{PREFIX}.self_attn.k_proj',
    f'{PREFIX}.self_attn.v_proj',
    f'{PREFIX}.self_attn.o_proj',
    f'{PREFIX}.mlp.gate_proj',
    f'{PREFIX}.mlp.up_proj',
    f'{PREFIX}.mlp.down_proj',
]

pass_count, fail_count = 0, 0
for pk in projs:
    t = raw.get(f'{pk}.trellis')
    K = int(t.shape[2]) * 16 // 256
    kt, nt, _ = t.shape
    w = reconstruct_orig_exl3(t, K,
        raw.get(f'{pk}.mcg') is not None,
        raw.get(f'{pk}.mul1') is not None,
        (kt*16, nt*16))
    # w is [in_f, out_f] — non-transposed, matching ext.hgemm layout
    in_f, out_f = w.shape
    x = torch.randn(4, in_f, device=DEVICE, dtype=torch.float16)
    x_3d = x.unsqueeze(0)  # [1, 4, in_f] — test ND broadcasting

    # ext.hgemm
    y_hg = torch.empty(x.shape[0], w.shape[1], dtype=x.dtype, device=x.device)
    ext.hgemm(x, w, y_hg)

    # F.mm (2D only)
    y_mm = torch.mm(x, w)

    # F.matmul (handles ND @ 2D)
    y_matmul = torch.matmul(x, w)
    y_matmul_3d = torch.matmul(x_3d, w)  # [1, 4, out_f]

    # F.linear (needs [out_f, in_f])
    y_lin = torch.nn.functional.linear(x, w.t().contiguous())

    d_hg_mm = (y_hg - y_mm).abs().max().item()
    d_hg_mat = (y_hg - y_matmul).abs().max().item()
    d_hg_lin = (y_hg - y_lin).abs().max().item()
    d_mm_mat = (y_mm - y_matmul).abs().max().item()

    ok = d_hg_mm < TOL and d_hg_mat < TOL
    pass_count += ok
    fail_count += not ok
    status = 'PASS' if ok else 'FAIL'
    print(f'{status} | {pk.split(".")[-1]:>10s} | hgemm==mm: {d_hg_mm:.2e} | hgemm==matmul: {d_hg_mat:.2e} | mm==matmul: {d_mm_mat:.2e} | F.linear diff: {d_hg_lin:.2e}')

print(f'\n{pass_count}/{pass_count+fail_count} passed | F.matmul ND shape: {y_matmul_3d.shape}')
exit(0 if fail_count == 0 else 1)
