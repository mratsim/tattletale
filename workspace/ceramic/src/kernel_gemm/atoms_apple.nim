## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Apple simdgroup MMA atom instantiations (bkGPU_TensorCore).
##
## The Apple GPU matrix-multiply-accumulate surface (all Apple GPUs, not
## M4-specific) is the Metal `simdgroup_multiply_accumulate` instruction on
## `simdgroup_float8x8` / `simdgroup_half8x8` fragments: one 8×8×8 tile per
## instruction, executed by a 32-lane SIMD group holding 2 values per lane
## per operand.
##
## Contraction rule: D = A·B + C, natural, no transposes, no operand swap:
##
##     D(r,c) = Σ_k A(r,k)·B(k,c)
##
## with A in (row = M, col = K) coordinates, B in (row = K, col = N) and
## D/C in (row = M, col = N). llama.cpp stores B transposed in threadgroup
## memory and passes it first; with row-major (K, N) B data the natural
## operand order (A first, B second) is correct.
##
## Fragment layout: each lane holds two adjacent elements of one fragment
## row. With qid = lane div 4:
##
##     fm = (qid and 4) + ((lane div 2) mod 4)     row 0..7
##     fn = (qid and 2) * 2 + (lane mod 2) * 2     col 0,2,4,6
##
## the lane holds (fm, fn) as element 0 and (fm, fn + 1) as element 1.
## This is the mapping MLX's `BaseMMAFrag<T,8,8>::get_coord` returns
## (`mlx/backend/metal/kernels/steel/gemm/mma.h`).
##
## simdgroup_load/store stride: the stride argument is the source row
## length, not the tile width. An 8×8 fragment of a 16-wide row-major
## matrix needs stride 16. llama.cpp's `ggml-metal.metal` keeps stride 8 by
## staging 8×8 blocks contiguously in threadgroup memory.
##
## Consumer: on Metal, `gemm_mma` (mma_dispatch.nim) emits the
## `simdgroup_multiply_accumulate` intrinsic for these atoms, and the MSL
## printer lowers the fragment tensors (`make_fragment_A/B/C`) to
## `simdgroup_float8x8` / `simdgroup_half8x8` with `simdgroup_load`/`store`
## gathers (see the simdgroup fragment ops in kernel_copy_gpu and
## kernel_fillwith_gpu).

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../atoms

# ═════════════════════════════════════════════════════════════════════════
#  Reusable layout aliases
# ═════════════════════════════════════════════════════════════════════════

const Apple8x8_AC_Layout* = make_layout(((2, 2, 2, 2, 2), 2), ((16, 1, 2, 32, 4), 8))
  ## (T32, V2) → (M8, K8) / (M8, N8) — A and C fragments of the 8×8×8 atoms,
  ## col-major offset m + 8·n, V stepping along K (A) or N (C).

const Apple8x8_B_Layout* = make_layout(((2, 2, 2, 2, 2), 2), ((2, 8, 16, 4, 32), 1))
  ## (T32, V2) → (N8, K8) — B fragment of the 8×8×8 atoms, col-major offset
  ## n + 8·k, V stepping along N.

# ═════════════════════════════════════════════════════════════════════════
#  Atoms
# ═════════════════════════════════════════════════════════════════════════

const APPLE_8x8x8_F32* = MmaAtom[
    typeof(Apple8x8_AC_Layout), typeof(Apple8x8_B_Layout), typeof(Apple8x8_AC_Layout)
  ](
    name: "APPLE_8x8x8_F32",
    mnk: (m: 8, n: 8, k: 8),
    aType: mdtF32, bType: mdtF32, cType: mdtF32,
    kind: bkGPU_TensorCore,
    instr: "simdgroup_multiply_accumulate",
    aLayout: Apple8x8_AC_Layout,
    bLayout: Apple8x8_B_Layout,
    cLayout: Apple8x8_AC_Layout,
  )
  ## f32 operands and accumulator on `simdgroup_float8x8` fragments.

const APPLE_8x8x8_F16* = MmaAtom[
    typeof(Apple8x8_AC_Layout), typeof(Apple8x8_B_Layout), typeof(Apple8x8_AC_Layout)
  ](
    name: "APPLE_8x8x8_F16",
    mnk: (m: 8, n: 8, k: 8),
    aType: mdtF16, bType: mdtF16, cType: mdtF32,
    kind: bkGPU_TensorCore,
    instr: "simdgroup_multiply_accumulate",
    aLayout: Apple8x8_AC_Layout,
    bLayout: Apple8x8_B_Layout,
    cLayout: Apple8x8_AC_Layout,
  )
  ## f16 operands, f32 accumulator on `simdgroup_half8x8` fragments.
