## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## The MMA atom catalog: every GPU MMA that exists, declared to
## `declareAtoms:` (h_configgen.nim).
##
## Glossary:
## - MMA:
##     Fused matrix multiply-accumulate instruction,
##     C <- A·B + C with C the accumulator.
## - Atom:
##     one MMA instruction descriptor
##     - MNK tile,
##     - operand datatypes,
##     - A mapping each thread's registers to tile elements.
## - Tile: the element block one atom computes per invocation.
##     - A is (M, K)
##     - B is (N, K)
##     - C is (M, N)
## - Fragment: the register values a thread holds for one operand tile, in the order the instruction expects.
## - T: the atom's threads.
## - V: the values each thread holds in registers.
##
## Hardware schema (m16n8k8 tf32 atom as a tiled GEMM, C = A·B):
##   each cell is the owning (thread, value):
##   - T = the atom's threads (0..31),
##   - v = the value's index in that thread's registers.
##     v is local to the thread:
##     T00.v0 and T01.v0 are different elements in different threads.
##   The k axis aligns. A's k column pairs with B's k row.
##
##                                                                   ┌──────────────────────────────────────────────────────────────┐
##                                                                   │ B (8, 8) — rows k, columns n                                 │
##                                                                   │ k=0  T00.v0 T04.v0 T08.v0 T12.v0 T16.v0 T20.v0 T24.v0 T28.v0 │
##                                                                   │ k=1  T01.v0 T05.v0 T09.v0 T13.v0 T17.v0 T21.v0 T25.v0 T29.v0 │
##                                                                   │ k=2  T02.v0 T06.v0 T10.v0 T14.v0 T18.v0 T22.v0 T26.v0 T30.v0 │
##                                                                   │ k=3  T03.v0 T07.v0 T11.v0 T15.v0 T19.v0 T23.v0 T27.v0 T31.v0 │
##                                                                   │ k=4  T00.v1 T04.v1 T08.v1 T12.v1 T16.v1 T20.v1 T24.v1 T28.v1 │
##                                                                   │ k=5  T01.v1 T05.v1 T09.v1 T13.v1 T17.v1 T21.v1 T25.v1 T29.v1 │
##                                                                   │ k=6  T02.v1 T06.v1 T10.v1 T14.v1 T18.v1 T22.v1 T26.v1 T30.v1 │
##                                                                   │ k=7  T03.v1 T07.v1 T11.v1 T15.v1 T19.v1 T23.v1 T27.v1 T31.v1 │
##                                                                   └──────────────────────────────────────────────────────────────┘
##
## ┌───────────────────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────────────────┐
## │ A (16, 8) — rows m, columns k                                 │ │ C (16, 8) — rows m, columns n                                │
## │ m=0  T00.v0 T01.v0 T02.v0 T03.v0 T00.v2 T01.v2 T02.v2 T03.v2  │ │ m=0  T00.v0 T00.v1 T01.v0 T01.v1 T02.v0 T02.v1 T03.v0 T03.v1 │
## │ m=1  T04.v0 T05.v0 T06.v0 T07.v0 T04.v2 T05.v2 T06.v2 T07.v2  │ │ m=1  T04.v0 T04.v1 T05.v0 T05.v1 T06.v0 T06.v1 T07.v0 T07.v1 │
## │ m=2  T08.v0 T09.v0 T10.v0 T11.v0 T08.v2 T09.v2 T10.v2 T11.v2  │ │ m=2  T08.v0 T08.v1 T09.v0 T09.v1 T10.v0 T10.v1 T11.v0 T11.v1 │
## │ m=3  T12.v0 T13.v0 T14.v0 T15.v0 T12.v2 T13.v2 T14.v2 T15.v2  │ │ m=3  T12.v0 T12.v1 T13.v0 T13.v1 T14.v0 T14.v1 T15.v0 T15.v1 │
## │ m=4  T16.v0 T17.v0 T18.v0 T19.v0 T16.v2 T17.v2 T18.v2 T19.v2  │ │ m=4  T16.v0 T16.v1 T17.v0 T17.v1 T18.v0 T18.v1 T19.v0 T19.v1 │
## │ m=5  T20.v0 T21.v0 T22.v0 T23.v0 T20.v2 T21.v2 T22.v2 T23.v2  │ │ m=5  T20.v0 T20.v1 T21.v0 T21.v1 T22.v0 T22.v1 T23.v0 T23.v1 │
## │ m=6  T24.v0 T25.v0 T26.v0 T27.v0 T24.v2 T25.v2 T26.v2 T27.v2  │ │ m=6  T24.v0 T24.v1 T25.v0 T25.v1 T26.v0 T26.v1 T27.v0 T27.v1 │
## │ m=7  T28.v0 T29.v0 T30.v0 T31.v0 T28.v2 T29.v2 T30.v2 T31.v2  │ │ m=7  T28.v0 T28.v1 T29.v0 T29.v1 T30.v0 T30.v1 T31.v0 T31.v1 │
## │ m=8  T00.v1 T01.v1 T02.v1 T03.v1 T00.v3 T01.v3 T02.v3 T03.v3  │ │ m=8  T00.v2 T00.v3 T01.v2 T01.v3 T02.v2 T02.v3 T03.v2 T03.v3 │
## │ m=9  T04.v1 T05.v1 T06.v1 T07.v1 T04.v3 T05.v3 T06.v3 T07.v3  │ │ m=9  T04.v2 T04.v3 T05.v2 T05.v3 T06.v2 T06.v3 T07.v2 T07.v3 │
## │ m=10 T08.v1 T09.v1 T10.v1 T11.v1 T08.v3 T09.v3 T10.v3 T11.v3  │ │ m=10 T08.v2 T08.v3 T09.v2 T09.v3 T10.v2 T10.v3 T11.v2 T11.v3 │
## │ m=11 T12.v1 T13.v1 T14.v1 T15.v1 T12.v3 T13.v3 T14.v3 T15.v3  │ │ m=11 T12.v2 T12.v3 T13.v2 T13.v3 T14.v2 T14.v3 T15.v2 T15.v3 │
## │ m=12 T16.v1 T17.v1 T18.v1 T19.v1 T16.v3 T17.v3 T18.v3 T19.v3  │ │ m=12 T16.v2 T16.v3 T17.v2 T17.v3 T18.v2 T18.v3 T19.v2 T19.v3 │
## │ m=13 T20.v1 T21.v1 T22.v1 T23.v1 T20.v3 T21.v3 T22.v3 T23.v3  │ │ m=13 T20.v2 T20.v3 T21.v2 T21.v3 T22.v2 T22.v3 T23.v2 T23.v3 │
## │ m=14 T24.v1 T25.v1 T26.v1 T27.v1 T24.v3 T25.v3 T26.v3 T27.v3  │ │ m=14 T24.v2 T24.v3 T25.v2 T25.v3 T26.v2 T26.v3 T27.v2 T27.v3 │
## │ m=15 T28.v1 T29.v1 T30.v1 T31.v1 T28.v3 T29.v3 T30.v3 T31.v3  │ │ m=15 T28.v2 T28.v3 T29.v2 T29.v3 T30.v2 T30.v3 T31.v2 T31.v3 │
## └───────────────────────────────────────────────────────────────┘ └──────────────────────────────────────────────────────────────┘
##
## T00's register values: A v0..v3 = (m, k) (0,0) (8,0) (0,4) (8,4).
## B v0..v1 = (k, n) (0,0) (4,0).
## C v0..v3 = (m, n) (0,0) (0,1) (8,0) (8,1).
## Each C value sums A(m,k)·B(k,n) over the 8 k entries. The k terms live in
## different threads (T00 holds only k ∈ {0, 4} of A and B), and the
## instruction sums them into the owning thread's register.
##
## Catalog: 4 universal FMA atoms (the 1×1×1 scalar fallback and the
## 8×8×8 cross-lane shuffle atoms on f32/f16/bf16), 3 Apple
## simdgroup atoms, 5 NVIDIA tensor-core atoms (SM80/SM89). The CPU
## atoms (AMX, SIMD ukernels) are not declared.
# TODO: pending the CPU-atom registry at CPU-merge time.

import ../int_tuples
import ../layouts
import ../layout_constructors
import ./h_configgen

# ═════════════════════════════════════════════════════════════════════════
#  Reusable layout aliases
# ═════════════════════════════════════════════════════════════════════════

const Universal8x8_AC_Layout* = make_layout(((2, 2, 2, 2, 2), 2), ((16, 1, 2, 32, 4), 8))
  ## (T32, V2) → (M8, K8) / (M8, N8): A and C fragments of the 8×8×8 atoms,
  ## col-major offset m + 8·n, V stepping along K (A) or N (C).
  ## A and C share one layout so an accumulator feeds the next mma's A
  ## operand with zero movement (the attention S→A handoff needs no
  ## redistribution). B strides along N, keeping each lane's two B values
  ## inside its own k row.

const Universal8x8_B_Layout* = make_layout(((2, 2, 2, 2, 2), 2), ((2, 8, 16, 4, 32), 1))
  ## (T32, V2) → (N8, K8): B fragment of the 8×8×8 atoms, col-major offset
  ## n + 8·k, V stepping along N.

const Apple8x8_AC_Layout* = make_layout(((2, 2, 2, 2, 2), 2), ((16, 1, 2, 32, 4), 8))
  ## (T32, V2) → (M8, K8) / (M8, N8) — A and C fragments of the Apple
  ## simdgroup atoms, col-major offset m + 8·n, V stepping along K (A) or N (C).

const Apple8x8_B_Layout* = make_layout(((2, 2, 2, 2, 2), 2), ((2, 8, 16, 4, 32), 1))
  ## (T32, V2) → (N8, K8) — B fragment of the Apple simdgroup atoms,
  ## col-major offset n + 8·k, V stepping along N.

const
  SM80_16x8_Row* = make_layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    ## (T32,V4) → (M16,N8) — C fragment of the m16n8k{8,16,32} f16·bf16·tf32·int8·fp8 atoms
  SM80_8x8_Row* = make_layout(((4, 8), 2), ((16, 1), 8))
    ## (T32,V2) → (M8,K8) — B fragment of m16n8k8
  SM80_16x8x8_A_TF32* = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    ## (T32,V4) → (M16,K8) — A fragment of m16n8k8 tf32
  SM80_16x8x8_B_TF32* = make_layout(((4, 8), 2), ((8, 1), 32))
    ## (T32,V2) → (N8,K8) — B fragment of m16n8k8 tf32
  SM80_16x8x16_A* = make_layout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    ## (T32,V8) → (M16,K16) — A fragment of m16n8k16 f16·bf16,
    ## the tensor-layouts reference transcription (atoms_nv.py)
  SM80_16x8x16_B* = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    ## (T32,V4) → (N8,K16) — B fragment of m16n8k16 f16·bf16
  SM80_16x8x32_A* = make_layout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    ## (T32,V16) → (M16,K32) — A fragment of m16n8k32 int8·fp8
  SM80_16x8x32_B* = make_layout(((4, 8), (4, 2)), ((32, 1), (8, 128)))
    ## (T32,V8) → (N8,K32) — B fragment of m16n8k32 int8·fp8

# ═════════════════════════════════════════════════════════════════════════
#  The atoms
# ═════════════════════════════════════════════════════════════════════════
#
#  vpt is the A fragment's values per thread (the B and C fragments' V
#  derive from their layouts via valuesPerThread). threadCount is 32
#  for every multi-lane atom, 1 for the scalar 1×1×1 fallback.

declareAtoms:
  # Universal FMA atoms: the gemm_atom scalar fallback (1×1×1) and the
  # cross-lane shuffle atoms. instr "" — plain arithmetic, no mnemonic.
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

  # Apple simdgroup atoms: the Metal simdgroup_multiply_accumulate intrinsic
  # on simdgroup_float8x8 / simdgroup_half8x8 fragments. Contraction is the
  # natural D = A·B + C, no transposes, no operand swap: A is (M, K), B is
  # (K, N) — llama.cpp stores B transposed and passes it first; with row-major
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

  # NVIDIA tensor-core atoms: the mma.sync extended-asm path. The fp8
  # atom shares the m16n8k32 int8 layouts.
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
