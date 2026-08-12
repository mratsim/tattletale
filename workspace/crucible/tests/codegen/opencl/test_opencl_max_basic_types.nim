## OpenCL: `max` on basic types (float32, uint32, int) — codegen inspection + run
##
## max is an "ambiguous builtin": both Nim and the GPU languages define it.
## Crucible registers the compiler-magic overloads name-only (forwarded to the
## backend's native max) and body-parses the non-magic ones like +/*:
##   - max(float32, float32) -> system's `{.inline.}` overload (no magic) ->
##     parsed body `if y <= x or y != y: x else: y` -> device fn with a ternary
##   - max(uint32, uint32)   -> generic `max*[T: not SomeFloat]` (no magic) ->
##     same: parsed body -> device fn with a ternary
##   - max(int, int)         -> magic MaxI -> plain `max` call (native)
##
## The fold pass (foldMaxMinToBuiltins) rewrites the ternary to the backend
## native max. On OpenCL C the integer `max`/`min` builtins are INTEGER-ONLY
## (spec §6.12.3) — floats require `fmax`/`fmin` (§6.12.2). The fold must
## lower float max/min to `fmax`/`fmin`: OpenCL C's `max` on float args
## compiles but applies integer semantics to the float bit patterns, so the
## result is silently wrong (e.g. 0.0 instead of 9.0).
##
## This file echoes the emitted OpenCL C so the codegen is visible, then runs
## the kernel and verifies the values.

import std/[unittest, strutils]
import workspace/crucible

# ── kernel ───────────────────────────────────────────────────────────────
# OpenCL binds kernel args in order: inputs first, output LAST.
const kernelCode = opencl:
  proc maxBasic(dyn: ptr UncheckedArray[float32];
                res: ptr UncheckedArray[float32]) {.global.} =
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
    # float32 min — same body-parsed path as max (fmin on OpenCL)
    res[6] = min(x, y)              # dynamic operands
    res[7] = min(3.5'f32, 7.25'f32) # literals
    # uint32 min — generic not-SomeFloat overload
    res[8] = float32(min(uint32(x), uint32(y)))

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelCode   # show the emitted OpenCL C

  suite "OpenCL — max on basic types":

    test "float32/uint32/int max compile, run, and produce the right values":
      # The fold must emit fmax/fmin for OpenCL floats — max/min are
      # integer-only there and silently compute wrong values on float args.
      check kernelCode.contains("fmax")
      check kernelCode.contains("fmin")

      var dynArr: array[2, float32] = [2.5'f32, 9.0'f32]
      var engine = bkOpenCL.init()
      engine.ingest(kernelCode)
      var out32: array[9, float32]
      engine.run("maxBasic", out32, (dynArr))
      check out32[0] == 9.0    # max(2.5, 9.0)
      check out32[1] == 7.25   # max(3.5, 7.25)
      check out32[2] == 9.0    # max(9.0, 2.5)
      check out32[3] == 9.0    # max(2, 9)
      check out32[4] == 7.0    # max(3, 7)
      check out32[5] == 9.0    # max(2, 9)
      check out32[6] == 2.5    # min(2.5, 9.0)
      check out32[7] == 3.5    # min(3.5, 7.25)
      check out32[8] == 2.0    # min(2, 9)

when isMainModule:
  runTest()
