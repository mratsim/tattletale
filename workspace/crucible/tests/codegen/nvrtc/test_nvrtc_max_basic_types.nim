## NVRTC: `max` on basic types (float32, uint32, int) — codegen inspection + run
##
## max is an "ambiguous builtin": both Nim and the GPU languages define it.
## Crucible registers the compiler-magic overloads name-only (forwarded to the
## backend's native max) and body-parses the non-magic ones like +/*:
##   - max(float32, float32) -> system's `{.inline.}` overload (no magic) ->
##     parsed body `if y <= x or y != y: x else: y` -> device fn with a ternary
##   - max(uint32, uint32)   -> generic `max*[T: not SomeFloat]` (no magic) ->
##     same: parsed body -> device fn with a ternary
##   - max(int, int)         -> magic MaxI -> plain `max` call (CUDA native)
##
## This file echoes the emitted CUDA so the codegen is visible, then runs the
## kernel and verifies the values.

import std/[unittest]
import workspace/crucible

# ── kernel ───────────────────────────────────────────────────────────────
# Output buffer MUST be the first kernel param (the harness prepends res).
const kernelCode = cuda:
  proc maxBasic(res: ptr UncheckedArray[float32];
                dyn: ptr UncheckedArray[float32]) {.global.} =
    let x = dyn[0]
    let y = dyn[1]
    # float32 — inline (non-magic) overload -> body-parsed device fn
    res[0] = max(x, y)              # dynamic operands
    res[1] = max(3.5'f32, 7.25'f32) # literals
    res[2] = max(y, x)              # swapped order
    # uint32 — generic not-SomeFloat overload -> body-parsed device fn
    res[3] = float32(max(uint32(x), uint32(y)))
    res[4] = float32(max(3'u32, 7'u32))
    # int — magic MaxI -> plain native max call
    res[5] = float32(max(int(x), int(y)))

echo kernelCode   # show the emitted CUDA

suite "NVRTC — max on basic types":

  test "float32/uint32/int max compile, run, and produce the right values":
    var buf: array[6, float32]
    var dynArr: array[2, float32] = [2.5'f32, 9.0'f32]
    var engine = bkCuda.init()
    engine.ingest(kernelCode)
    engine.run<<(1, 1)>>("maxBasic", buf, (dynArr,))
    check buf[0] == 9.0    # max(2.5, 9.0)
    check buf[1] == 7.25   # max(3.5, 7.25)
    check buf[2] == 9.0    # max(9.0, 2.5)
    check buf[3] == 9.0    # max(2, 9)
    check buf[4] == 7.0    # max(3, 7)
    check buf[5] == 9.0    # max(2, 9)
