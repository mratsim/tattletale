## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## NVIDIA tensor-core MMA atom instantiations (bkGPU_TensorCore).
##
## Layouts transcribed from facebookresearch/tensor-layouts
## `src/tensor_layouts/atoms_nv.py` (which are themselves validated by oracle
## tests against CUTLASS C++). (T, V) → col-major offset in the operand tile.

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../atoms

# ═════════════════════════════════════════════════════════════════════════
#  Reusable layout aliases (from atoms_nv.py:210-216 / mma_traits_sm80.hpp)
# ═════════════════════════════════════════════════════════════════════════

const
  SM80_16x8_Row* = make_layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    ## (T32,V4) → (M16,N8) — C fragment of m16n8k8/m16n8k16 f16·bf16·tf32
  SM80_8x8_Row* = make_layout(((4, 8), 2), ((16, 1), 8))
    ## (T32,V2) → (M8,K8) — B fragment of m16n8k8
  SM80_16x8x8_A_TF32* = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    ## (T32,V4) → (M16,K8) — A fragment of m16n8k8 tf32
  SM80_16x8x8_B_TF32* = make_layout(((4, 8), 2), ((8, 1), 32))
    ## (T32,V2) → (N8,K8) — B fragment of m16n8k8 tf32

# ═════════════════════════════════════════════════════════════════════════
#  Atoms
# ═════════════════════════════════════════════════════════════════════════

const SM80_16x8x8_F32TF32TF32F32_TN* = MmaAtom[
    typeof(SM80_16x8x8_A_TF32), typeof(SM80_16x8x8_B_TF32), typeof(SM80_16x8_Row)
  ](
    name: "SM80_16x8x8_F32TF32TF32F32_TN",
    mnk: (m: 16, n: 8, k: 8),
    aType: mdtTF32, bType: mdtTF32, cType: mdtF32,
    scaleMode: smNone, blockSize: 0,
    sfaType: mdtF32, sfbType: mdtF32,
    kind: bkGPU_TensorCore,
    instr: "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
    aLayout: SM80_16x8x8_A_TF32,
    bLayout: SM80_16x8x8_B_TF32,
    cLayout: SM80_16x8_Row,
    scaleVec: sv1X,
  )

# SM86 (consumer Ampere) runs the sm_80 mma.sync family unchanged — the
# tf32 m16n8k8 is the TF32 tensor-core kernel introduced with the Ampere
# generation (A100 = sm_80, consumer RTX 30 = sm_86). The atom is the
# same instruction, arch-labeled for the consumer-Ampere target.
const SM86_16x8x8_F32TF32TF32F32_TN* = MmaAtom[
    typeof(SM80_16x8x8_A_TF32), typeof(SM80_16x8x8_B_TF32), typeof(SM80_16x8_Row)
  ](
    name: "SM86_16x8x8_F32TF32TF32F32_TN",
    mnk: (m: 16, n: 8, k: 8),
    aType: mdtTF32, bType: mdtTF32, cType: mdtF32,
    scaleMode: smNone, blockSize: 0,
    sfaType: mdtF32, sfbType: mdtF32,
    kind: bkGPU_TensorCore,
    instr: "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
    aLayout: SM80_16x8x8_A_TF32,
    bLayout: SM80_16x8x8_B_TF32,
    cLayout: SM80_16x8_Row,
    scaleVec: sv1X,
  )
