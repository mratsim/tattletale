## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Universal software-mma atom instantiations (bk_FMA).
##
## The 8×8×8 atom whose A and C fragments share the Apple AC layout and
## whose per-atom `mma` is the explicit cross-lane shuffle reduction
## (tile_config): the portable spelling of what the M4 simdgroup
## multiply-accumulate lowers to the same reduction with explicit shuffles
## (16 FMA + 6 SHFF on the hardware vs 24 gathers + 16 FMAs). Because
## A and C share one fragment layout, an accumulator feeds the next mma's
## A operand with zero movement. The attention S→A handoff needs no
## redistribution. Runs on every backend that has subgroup shuffles
## (Metal simd_shuffle, OpenCL 2.0+ sub_group_shuffle, CUDA shfl).
##
## The Apple 8×8 fragment contract, per lane:
##   fm = (qid and 4) + ((lane div 2) mod 4)      fragment row 0..7
##   fn = (qid and 2)·2 + (lane mod 2)·2          fragment col 0,2,4,6
##   with qid = lane div 4 → the lane holds (fm, fn) and (fm, fn + 1).
## B uses the Apple B layout (V step along N instead of K).
##
## Based on CuTe `UniversalFMA` (CUTLASS `cute/arch/mma.hpp`), whose
## degenerate (1, 1, 1) scalar form is specialized here to the 8×8×8
## AC-contract atom.

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../atoms
import ./atoms_apple

const UNIVERSAL_FMA_F32* = MmaAtom[
    typeof(Apple8x8_AC_Layout), typeof(Apple8x8_B_Layout), typeof(Apple8x8_AC_Layout)
  ](
    name: "UNIVERSAL_FMA_F32",
    mnk: (m: 8, n: 8, k: 8),
    aType: mdtF32, bType: mdtF32, cType: mdtF32,
    kind: bk_FMA,
    instr: "",
    aLayout: Apple8x8_AC_Layout,
    bLayout: Apple8x8_B_Layout,
    cLayout: Apple8x8_AC_Layout,
  )
  ## 8×8×8 software-mma atom: 32 lanes × 2 values per operand, float32
  ## operands and accumulator. A and C share the Apple AC layout, so
  ## C-role == A-role and an accumulator feeds the next mma's A operand
  ## without movement.

const UNIVERSAL_FMA_F16* = MmaAtom[
    typeof(Apple8x8_AC_Layout), typeof(Apple8x8_B_Layout), typeof(Apple8x8_AC_Layout)
  ](
    name: "UNIVERSAL_FMA_F16",
    mnk: (m: 8, n: 8, k: 8),
    aType: mdtF16, bType: mdtF16, cType: mdtF32,
    kind: bk_FMA,
    instr: "",
    aLayout: Apple8x8_AC_Layout,
    bLayout: Apple8x8_B_Layout,
    cLayout: Apple8x8_AC_Layout,
  )
  ## 8×8×8 software-mma atom: float16 operands widened to the float32
  ## accumulator, the fp16→fp32 form.

const UNIVERSAL_FMA_BF16* = MmaAtom[
    typeof(Apple8x8_AC_Layout), typeof(Apple8x8_B_Layout), typeof(Apple8x8_AC_Layout)
  ](
    name: "UNIVERSAL_FMA_BF16",
    mnk: (m: 8, n: 8, k: 8),
    aType: mdtBF16, bType: mdtBF16, cType: mdtF32,
    kind: bk_FMA,
    instr: "",
    aLayout: Apple8x8_AC_Layout,
    bLayout: Apple8x8_B_Layout,
    cLayout: Apple8x8_AC_Layout,
  )
  ## 8×8×8 software-mma atom: bfloat16 operands widened to the float32
  ## accumulator, the bf16→fp32 form.
