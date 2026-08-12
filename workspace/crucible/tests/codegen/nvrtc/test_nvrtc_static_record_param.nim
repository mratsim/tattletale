## NVRTC: func with a `static` record param called from a `cuda:` kernel
## Run with: nim cpp -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##   workspace/crucible/tests/codegen/nvrtc/test_nvrtc_static_record_param.nim
##
## A `static T` param of record type is compile-time only: the instantiated
## proc type carries the VALUE in place of the type (e.g. `static MiniAtom`
## -> `MiniAtom(dtype: ..., k: ...)`), which has no CUDA value and whose
## fields (enums) resolveType cannot lower. Crucible must drop the param from
## the signature and the matching argument at the call site, like typedesc
## params. Regression test: without the fix this file fails to compile inside
## the `cuda:` macro ("Type: ntyEnum not supported yet").
import workspace/crucible

type
  MiniDType = enum
    mdtF32, mdtTF32

  MiniAtom = object
    ## minimal stand-in for a compiler atom record (record with an enum field)
    dtype: MiniDType
    k: int

func mini_ukernel(mma: static MiniAtom, cFrag: var array[4, float32]) {.inline.} =
  ## the static record param's FIELDS are usable directly in the body — the
  ## field access is evaluated at compile time and emitted as a literal
  cFrag[0] = cFrag[0] + float32(mma.k)

const kernelCode = cuda:
  proc staticParamKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var c = [0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    mini_ukernel(MiniAtom(dtype: mdtTF32, k: 8), c)
    output[0] = c[0]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, float32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  engine.run("staticParamKernel", buf, ())
  echo "  [0]=", buf[0]
  doAssert buf[0] == 8.0'f32
  echo "  OK (test_nvrtc_static_record_param)"

when isMainModule:
  runTest()
