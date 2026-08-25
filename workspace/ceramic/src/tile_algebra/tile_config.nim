## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../hardware/h_configgen
import ../hardware/h_registry
import ../hardware/h_properties
import workspace/crucible

export h_configgen, h_registry, h_properties
# Re-exported so tile consumers keep one import surface (see tiles.nim).

# ═════════════════════════════════════════════════════════════════════════
#  getTileConfig: the per-backend, per-element-type atom mapping
# ═════════════════════════════════════════════════════════════════════════

func getTileConfig*(TOut, TIn: typedesc): static auto {.inline.} =
  when ccGetBackend() == ctMetal:
    when TOut is float32 and TIn is float16: UNIVERSAL_8x8x8_F32F16F16F32
    elif TOut is float32 and TIn is bfloat16: UNIVERSAL_8x8x8_F32BF16BF16F32
    elif TOut is float32 and TIn is float32: UNIVERSAL_8x8x8_F32F32F32F32
    else: {.error: "getTileConfig: no atom for (" & $TOut & ", " & $TIn &
      ") mix (fp32 accumulator with f16/bf16/f32 operands only)".}
  elif ccGetBackend() == ctCuda:
    when TOut is float32 and TIn is float16: UNIVERSAL_8x8x8_F32F16F16F32
    elif TOut is float32 and TIn is bfloat16: UNIVERSAL_8x8x8_F32BF16BF16F32
    elif TOut is float32 and TIn is float32: UNIVERSAL_8x8x8_F32F32F32F32
    else: {.error: "getTileConfig: no atom for (" & $TOut & ", " & $TIn &
      ") mix (fp32 accumulator with f16/bf16/f32 operands only)".}
  else:
    when TOut is float32 and TIn is float16: UNIVERSAL_8x8x8_F32F16F16F32
    elif TOut is float32 and TIn is bfloat16: UNIVERSAL_8x8x8_F32BF16BF16F32
    elif TOut is float32 and TIn is float32: UNIVERSAL_8x8x8_F32F32F32F32
    else: {.error: "getTileConfig: no atom for (" & $TOut & ", " & $TIn &
      ") mix (fp32 accumulator with f16/bf16/f32 operands only)".}

# ═════════════════════════════════════════════════════════════════════════
#  SubTile: the per-lane register fragment type
# ═════════════════════════════════════════════════════════════════════════

type
  SubTile*[A: static MmaAtom; T] = object
    ## One atom subtile's per-lane register fragment: one slot per value
    ## the atom assigns to a lane (two on the 8×8×8 atoms).
    frag*: array[A.getVpt(), T]
