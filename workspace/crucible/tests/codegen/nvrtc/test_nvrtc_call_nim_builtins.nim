## Tests that registerGenericInstOrExternalProc handles:
##   1. system.* called as nnkCall (inside for-loop) — magic builtin
##   2. User-defined `*` on struct called as nnkCall — non-magic pull-in
##
## Both must go through the same codegen path and not assert.
## Self-contained — no ceramic dependency.
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_call_nim_builtins.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc

# ── system.* as nnkCall with runtime args ──
# system.* has magic: MulI → collectProcAttributes returns empty
# → toGpuAst returns gpuDiscard (stored in ctx.builtins)
# → assert inst.isBuiltIn() fails because the custom isBuiltIn
#   only checks for "builtin"/"importc", not "magic"
# → fix: check NimGpuNumericOperators before reaching the assert

const callOpKernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    for i in 0 ..< 2:
      let x = `*`(i + 2, 3)
      C[i] = float32(x)

# ── Non-magic user-defined `*` called as nnkCall ──
# Must go through the same codegen path and succeed.

type Wrapper = object
  val: int

proc `*`(a, b: Wrapper): Wrapper =
  Wrapper(val: a.val * b.val)

const wrapperKernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    for i in 0 ..< 2:
      let a = Wrapper(val: i + 3)
      let b = Wrapper(val: 4)
      let c = `*`(a, b)
      C[i] = float32(c.val)

suite "NVRTC - call nim builtins":

  test "system.`*` as nnkCall compiles and runs":
    var buf: array[2, float32]
    var nv = initNvrtc(callOpKernel)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
    check buf[0] == 6.0'f32
    check buf[1] == 9.0'f32

  test "user-defined `*` on struct as nnkCall compiles and runs":
    var buf: array[2, float32]
    var nv = initNvrtc(wrapperKernel)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
    check buf[0] == 12.0'f32
    check buf[1] == 16.0'f32
