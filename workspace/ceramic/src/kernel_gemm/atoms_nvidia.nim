## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## NVIDIA tensor-core MMA atom instantiations (bkGPU_TensorCore).

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../atoms

# ═════════════════════════════════════════════════════════════════════════
#  Reusable layout aliases
# ═════════════════════════════════════════════════════════════════════════

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
#  Atoms
# ═════════════════════════════════════════════════════════════════════════

const SM80_16x8x8_F32TF32TF32F32_TN* = MmaAtom[
    typeof(SM80_16x8x8_A_TF32), typeof(SM80_16x8x8_B_TF32), typeof(SM80_16x8_Row)
  ](
    name: "SM80_16x8x8_F32TF32TF32F32_TN",
    mnk: (m: 16, n: 8, k: 8),
    aType: mdtTF32, bType: mdtTF32, cType: mdtF32,
    kind: bkGPU_TensorCore,
    instr: "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
    aLayout: SM80_16x8x8_A_TF32,
    bLayout: SM80_16x8x8_B_TF32,
    cLayout: SM80_16x8_Row,
  )
  ## tf32 operands, fp32 accumulator, the native m16n8k8 shape.

const SM80_16x8x16_F32BF16BF16F32_TN* = MmaAtom[
    typeof(SM80_16x8x16_A), typeof(SM80_16x8x16_B), typeof(SM80_16x8_Row)
  ](
    name: "SM80_16x8x16_F32BF16BF16F32_TN",
    mnk: (m: 16, n: 8, k: 16),
    aType: mdtBF16, bType: mdtBF16, cType: mdtF32,
    kind: bkGPU_TensorCore,
    instr: "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
    aLayout: SM80_16x8x16_A,
    bLayout: SM80_16x8x16_B,
    cLayout: SM80_16x8_Row,
  )
  ## bf16 operands, fp32 accumulator, the native m16n8k16 shape.

const SM80_16x8x16_F32F16F16F32_TN* = MmaAtom[
    typeof(SM80_16x8x16_A), typeof(SM80_16x8x16_B), typeof(SM80_16x8_Row)
  ](
    name: "SM80_16x8x16_F32F16F16F32_TN",
    mnk: (m: 16, n: 8, k: 16),
    aType: mdtF16, bType: mdtF16, cType: mdtF32,
    kind: bkGPU_TensorCore,
    instr: "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
    aLayout: SM80_16x8x16_A,
    bLayout: SM80_16x8x16_B,
    cLayout: SM80_16x8_Row,
  )
  ## f16 operands, fp32 accumulator — the same layouts as the bf16 atom.

const SM80_16x8x32_S32S8S8S32_TN* = MmaAtom[
    typeof(SM80_16x8x32_A), typeof(SM80_16x8x32_B), typeof(SM80_16x8_Row)
  ](
    name: "SM80_16x8x32_S32S8S8S32_TN",
    mnk: (m: 16, n: 8, k: 32),
    aType: mdtInt8, bType: mdtInt8, cType: mdtInt32,
    kind: bkGPU_TensorCore,
    instr: "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
    aLayout: SM80_16x8x32_A,
    bLayout: SM80_16x8x32_B,
    cLayout: SM80_16x8_Row,
  )
  ## int8 operands, int32 accumulator, the native m16n8k32 shape.

const SM89_16x8x32_F32E4M3E4M3F32_TN* = MmaAtom[
    typeof(SM80_16x8x32_A), typeof(SM80_16x8x32_B), typeof(SM80_16x8_Row)
  ](
    name: "SM89_16x8x32_F32E4M3E4M3F32_TN",
    mnk: (m: 16, n: 8, k: 32),
    aType: mdtFP8E4M3, bType: mdtFP8E4M3, cType: mdtF32,
    kind: bkGPU_TensorCore,
    instr: "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
    aLayout: SM80_16x8x32_A,
    bLayout: SM80_16x8x32_B,
    cLayout: SM80_16x8_Row,
  )
  ## fp8 (E4M3) operands, fp32 accumulator — the same layouts as the
  ## m16n8k32 int8 atom (the SM89 reference transcription).
