## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Matrix-Multiply-Accumulate (Tensor core) instruction catalog, declarative only.
##
## Glossary:
##
##   - MMA, C <- A·B + C, C the accumulator
##   - Atom, one instruction descriptor (MNK tile, operand datatypes, thread-to-element mapping)
##   - Tile, the element block one atom computes per invocation, A (M, K), B (N, K), C (M, N)
##
## Fragment, the register values a thread holds for one operand tile, in instruction order.
## T, the atom's threads. V, the values each thread holds in registers.

import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/int_tuples
import ./h_mma_configgen

# ═════════════════════════════════════════════════════════════════════════
#  Reusable layout aliases
# ═════════════════════════════════════════════════════════════════════════

const Universal8x8_AC_Layout* = make_layout(((2, 2, 2, 2, 2), 2), ((16, 1, 2, 32, 4), 8))
  ## (T32, V2) → (M8, K8) / (M8, N8), A and C fragments
  ## of the 8×8×8 atoms, col-major offset m + 8·n, V stepping along K (A) or N (C).
  ##
  ## A and C share one layout so an accumulator feeds the next mma's
  ## A operand with zero movement, the attention S→A handoff needs no redistribution.
  ##
  ## B strides along N, keeping each lane's two B values inside its own k row.

const Universal8x8_B_Layout* = make_layout(((2, 2, 2, 2, 2), 2), ((2, 8, 16, 4, 32), 1))
  ## (T32, V2) → (N8, K8), B fragment of the 8×8×8 atoms, col-major offset
  ## n + 8·k, V stepping along N.

const Apple8x8_AC_Layout* = make_layout(((2, 2, 2, 2, 2), 2), ((16, 1, 2, 32, 4), 8))
  ## (T32, V2) → (M8, K8) / (M8, N8), the A and C fragments of the Apple
  ## simdgroup atoms, col-major offset m + 8·n, V stepping along K (A) or N (C).

const Apple8x8_B_Layout* = make_layout(((2, 2, 2, 2, 2), 2), ((2, 8, 16, 4, 32), 1))
  ## (T32, V2) → (N8, K8), B fragment of the Apple simdgroup atoms,
  ## col-major offset n + 8·k, V stepping along N.

const
  SM80_16x8_Row* = make_layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    ## (T32,V4) → (M16,N8), C fragment of the m16n8k{8,16,32} f16·bf16·tf32·int8·fp8 atoms
  SM80_8x8_Row* = make_layout(((4, 8), 2), ((16, 1), 8))
    ## (T32,V2) → (M8,K8), B fragment of m16n8k8
  SM80_16x8x8_A_TF32* = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    ## (T32,V4) → (M16,K8), A fragment of m16n8k8 tf32
  SM80_16x8x8_B_TF32* = make_layout(((4, 8), 2), ((8, 1), 32))
    ## (T32,V2) → (N8,K8), B fragment of m16n8k8 tf32
  SM80_16x8x16_A* = make_layout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    ## (T32,V8) → (M16,K16), A fragment of m16n8k16 f16·bf16,
    ## the tensor-layouts reference transcription (atoms_nv.py)
  SM80_16x8x16_B* = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    ## (T32,V4) → (N8,K16), B fragment of m16n8k16 f16·bf16
  SM80_16x8x32_A* = make_layout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    ## (T32,V16) → (M16,K32), A fragment of m16n8k32 int8·fp8
  SM80_16x8x32_B* = make_layout(((4, 8), (4, 2)), ((32, 1), (8, 128)))
    ## (T32,V8) → (N8,K32), B fragment of m16n8k32 int8·fp8

# ═════════════════════════════════════════════════════════════════════════
#  The atoms
# ═════════════════════════════════════════════════════════════════════════
#
#  vpt is the A fragment's values per thread, the B and C fragments'
#  V derive from their layouts via valuesPerThread.
#  threadCount is 32 for every multi-lane atom, 1 for the 1×1×1 fallback.

declareAtoms:
  # Universal FMA atoms, the gemm_atom scalar fallback (1×1×1)
  # and the cross-lane shuffle atoms. instr "" is plain arithmetic, no mnemonic.
  atom UNIVERSAL_1x1x1_F32F32F32F32:
    m: 1
    n: 1
    k: 1
    vpt: 1
    threadCount: 1
    aLayout: make_layout((1, 1))
    bLayout: make_layout((1, 1))
    cLayout: make_layout((1, 1))
    instr: ""
  atom UNIVERSAL_8x8x8_F32F32F32F32:
    m: 8
    n: 8
    k: 8
    vpt: 2
    threadCount: 32
    aLayout: Universal8x8_AC_Layout
    bLayout: Universal8x8_B_Layout
    cLayout: Universal8x8_AC_Layout
    instr: ""
  atom UNIVERSAL_8x8x8_F32F16F16F32:
    m: 8
    n: 8
    k: 8
    vpt: 2
    threadCount: 32
    aLayout: Universal8x8_AC_Layout
    bLayout: Universal8x8_B_Layout
    cLayout: Universal8x8_AC_Layout
    instr: ""
  atom UNIVERSAL_8x8x8_F32BF16BF16F32:
    m: 8
    n: 8
    k: 8
    vpt: 2
    threadCount: 32
    aLayout: Universal8x8_AC_Layout
    bLayout: Universal8x8_B_Layout
    cLayout: Universal8x8_AC_Layout
    instr: ""

  # Apple simdgroup atoms, the Metal simdgroup_multiply_accumulate intrinsic
  # on simdgroup_float8x8 / simdgroup_half8x8 fragments.
  #
  # Contraction is the natural D = A·B + C, no transposes, no operand swap,
  # A is (M, K), B is (K, N).
  #
  # llama.cpp stores B transposed and passes it first, with row-major
  # (K, N) B data the natural operand order (A first, B second) is correct.
  atom APPLE_8x8x8_F32:
    m: 8
    n: 8
    k: 8
    vpt: 2
    threadCount: 32
    aLayout: Apple8x8_AC_Layout
    bLayout: Apple8x8_B_Layout
    cLayout: Apple8x8_AC_Layout
    instr: "simdgroup_multiply_accumulate"
    elem: "float"
  atom APPLE_8x8x8_F16:
    m: 8
    n: 8
    k: 8
    vpt: 2
    threadCount: 32
    aLayout: Apple8x8_AC_Layout
    bLayout: Apple8x8_B_Layout
    cLayout: Apple8x8_AC_Layout
    instr: "simdgroup_multiply_accumulate"
    elem: "half"
  atom APPLE_8x8x8_BF16:
    m: 8
    n: 8
    k: 8
    vpt: 2
    threadCount: 32
    aLayout: Apple8x8_AC_Layout
    bLayout: Apple8x8_B_Layout
    cLayout: Apple8x8_AC_Layout
    instr: "simdgroup_multiply_accumulate"
    elem: "bfloat"

  # NVIDIA tensor-core atoms, the mma.sync extended-asm path.
  # The fp8 atom shares the m16n8k32 int8 layouts.
  atom SM80_16x8x8_F32TF32TF32F32_TN:
    m: 16
    n: 8
    k: 8
    vpt: 4
    threadCount: 32
    aLayout: SM80_16x8x8_A_TF32
    bLayout: SM80_16x8x8_B_TF32
    cLayout: SM80_16x8_Row
    instr: "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
  atom SM80_16x8x16_F32BF16BF16F32_TN:
    m: 16
    n: 8
    k: 16
    vpt: 8
    threadCount: 32
    aLayout: SM80_16x8x16_A
    bLayout: SM80_16x8x16_B
    cLayout: SM80_16x8_Row
    instr: "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
  atom SM80_16x8x16_F32F16F16F32_TN:
    m: 16
    n: 8
    k: 16
    vpt: 8
    threadCount: 32
    aLayout: SM80_16x8x16_A
    bLayout: SM80_16x8x16_B
    cLayout: SM80_16x8_Row
    instr: "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  atom SM80_16x8x32_S32S8S8S32_TN:
    m: 16
    n: 8
    k: 32
    vpt: 16
    threadCount: 32
    aLayout: SM80_16x8x32_A
    bLayout: SM80_16x8x32_B
    cLayout: SM80_16x8_Row
    instr: "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
  atom SM89_16x8x32_F32E4M3E4M3F32_TN:
    m: 16
    n: 8
    k: 32
    vpt: 16
    threadCount: 32
    aLayout: SM80_16x8x32_A
    bLayout: SM80_16x8x32_B
    cLayout: SM80_16x8_Row
    instr: "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
