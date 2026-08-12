## Test that `determineArrayLength` handles `nnkSym` with `allowArrayIdent=true`
## without crashing (CR 5 — previously called `getImpl.intVal` unconditionally
## before the `allowArrayIdent` check, overwriting the result).
##
## An external constant (`SPONGE_WIDTH`) used as array length triggers the
## `nnkSym` path in `determineArrayLength`. With `allowArrayIdent=true`,
## the function must return -1 (gray area) without crashing on `getImpl`.
##
## Run with:
##   nim c -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_external_array_len.nim
import std/strformat
import workspace/crucible

const SPONGE_WIDTH = 16

type
  BigInt = object
    limbs: array[SPONGE_WIDTH, uint32]

proc initBigInt(v: uint32): BigInt {.device.} =
  for i in 0 ..< SPONGE_WIDTH:
    result.limbs[i] = v

proc bigIntSum(b: BigInt): uint32 {.device.} =
  for i in 0 ..< SPONGE_WIDTH:
    result += b.limbs[i]

const kernelCode = cuda:
  proc testKernel(outp: ptr UncheckedArray[uint32]) {.global.} =
    let b = initBigInt(1'u32)
    outp[0] = bigIntSum(b)

# If compilation succeeds, the nnkSym+allowArrayIdent=true path
# did not crash (getImpl.intVal was not called unconditionally).
proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"

  echo "  OK — external array length resolution (nnkSym + allowArrayIdent=true) compiles without crash"

when isMainModule:
  runTest()
