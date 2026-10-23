## SPIR-V codegen refusals: kernel-shape and push-constant-type rejections
##
## Test that the spirv macro refuses unsupported kernel shapes at compile time.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/t_spirv_refusals.nim

import std/[os, osproc, strutils]
import workspace/crucible

# External types and device fns live outside the `spirv:` blocks,
# the negative kernels below reference them by symbol

type
  C2Pair = object
    a: uint32
    b: uint32

template noGlobalKernelBlock() =
  let k = spirv:
    proc flagSet(output: ptr uint32, flag: bool) =
      if flag:
        output[] = 1'u32
      else:
        output[] = 0'u32
  discard k

proc fact(n: int32): int32 {.device.} =
  if n <= 1:
    result = 1
  else:
    result = n * fact(n - 1)

# One block per major refusal guard. Guards covered: kernel/param shape
# (shared/SMEM param, void ptr param, byRef scalar param), control flow
# (return in loop body, every-branch-returns, recursive device fn), stores
# (memcpy assign, BDA deref store, float comparison).
# A block that compiles means the guard is gone: that is a live bug.

template smemParamBlock() =
  let k = spirv:
    proc smemKernel(output: ptr UncheckedArray[uint32]) {.global.} =
      var scratch {.smem.}: array[64, uint32]
      scratch[0] = 1'u32
      output[0] = scratch[0]
  discard k

template voidPtrParamBlock() =
  let k = spirv:
    proc voidPtrKernel(p: pointer) {.global.} =
      discard
  discard k

template byRefScalarParamBlock() =
  let k = spirv:
    proc byRefKernel(x: var uint32) {.global.} =
      x = 1'u32
  discard k

template returnInLoopBlock() =
  let k = spirv:
    proc loopReturnKernel(x: ptr UncheckedArray[uint32], n: int32) {.global.} =
      for i in 0 ..< n:
        return
  discard k

template everyBranchReturnsBlock() =
  let k = spirv:
    proc branchReturnKernel(flag: int32) {.global.} =
      if flag > 0:
        return
      else:
        return
  discard k

template memcpyAssignBlock() =
  let k = spirv:
    proc memcpyAssignKernel(output: ptr UncheckedArray[float32]) {.global.} =
      var acc: array[2, float32] = [1.0'f32, 2.0'f32]
      var other: array[2, float32] = [5.0'f32, 6.0'f32]
      acc = other
      output[0] = acc[0]
  discard k

template bdaDerefStoreBlock() =
  let k = spirv:
    proc derefStoreKernel(output: ptr uint32) {.global.} =
      output[] = 1'u32
  discard k

template recursiveDeviceFnBlock() =
  let k = spirv:
    proc factKernel(output: ptr UncheckedArray[int32], n: int32) {.global.} =
      output[0] = fact(n)
  discard k

template floatComparisonBlock() =
  let k = spirv:
    proc floatCmpKernel(output: ptr UncheckedArray[float32], x: ptr UncheckedArray[float32], n: int32) {.global.} =
      for i in 0 ..< n:
        if x[i] > 0.5'f32:
          output[i] = 1.0'f32
  discard k

const boolRefusalMessage =
  "push-constant member bool, OpTypeBool cannot appear in a push-constant block"

proc compileFromSource(src: string): tuple[output: string, exitCode: int] =
  ## Compiles a generated check file against the workspace/crucible
  ## import path, returns the compiler output for message assertions
  let testDir = currentSourcePath().parentDir()
  # tests/codegen/vulkan_via_spirv -> the repo root that carries workspace/
  let projectRoot = testDir / "../../../.."
  let tmpDir = testDir / "nimcache" / "refusal_checks"
  createDir(tmpDir)
  let checkPath = tmpDir / "check_bool_push_constant.nim"
  writeFile(checkPath, src)
  let cmd = "nim c --hints:off --warnings:off" &
            " --path:" & projectRoot.absolutePath &
            " --outdir:" & tmpDir & " --nimcache:" & tmpDir / "nimcache" &
            " " & checkPath
  result = execCmdEx(cmd)
  removeFile(checkPath)

proc runTest() =
  block noGlobalKernel:
    doAssert not compiles(noGlobalKernelBlock()),
      "a spirv: block without a {.global.} kernel must refuse to compile"

  block boolPushConstant:
    # compiles() cannot distinguish the documented refusal from any other
    # compile-time failure, the message is asserted on the compiler output
    let (outp, rc) = compileFromSource("""
import workspace/crucible
let k = spirv:
  proc flagSet(output: ptr uint32, flag: bool) {.global.} =
    if flag:
      output[] = 1'u32
    else:
      output[] = 0'u32
discard k
""")
    doAssert rc != 0, "a bool kernel parameter must refuse to compile, got: " & outp
    doAssert boolRefusalMessage in outp,
      "the bool push-constant refusal must name the OpTypeBool block defect, got: " & outp

  block smemParam:
    # {.smem.} locals refuse via the shared/constant-locals guard.
    # The asSMEM param guard stays defensive for shapes the macro cannot produce
    doAssert not compiles(smemParamBlock()),
      "a {.smem.} local must refuse to compile"

  block voidPtrParam:
    doAssert not compiles(voidPtrParamBlock()),
      "a void ptr kernel parameter must refuse to compile"

  block byRefScalarParam:
    # a var scalar param lowers to a device-address pointer param
    # the store then hits the BDA deref-store guard. Its message names
    # the store site
    doAssert not compiles(byRefScalarParamBlock()),
      "a byRef scalar kernel parameter must refuse to compile"

  block returnInLoop:
    # a return nested in a selection inside the loop body stays supported,
    # the continue target remains reachable. A bare loop-body return refuses
    doAssert not compiles(returnInLoopBlock()),
      "a return as the loop body's last statement must refuse to compile"

  block everyBranchReturns:
    doAssert not compiles(everyBranchReturnsBlock()),
      "an if-statement whose every branch returns must refuse to compile"

  block memcpyAssign:
    doAssert not compiles(memcpyAssignBlock()),
      "an array assign that requires a memcpy must refuse to compile"

  block bdaDerefStore:
    doAssert not compiles(bdaDerefStoreBlock()),
      "a deref store through a buffer device address must refuse to compile"

  block recursiveDeviceFn:
    doAssert not compiles(recursiveDeviceFnBlock()),
      "a recursive device function call must refuse to compile"

  block floatComparison:
    doAssert not compiles(floatComparisonBlock()),
      "a float comparison must refuse to compile"

when isMainModule:
  runTest()
  echo "t_spirv_refusals: all checks passed"
