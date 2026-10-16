## NVRTC: `var array` and `var openArray` params lower as `T*` so inline-asm
## operands reference elements, not the decayed whole-array pointer.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_nvrtc_var_array_asm.nim
import workspace/crucible

func addAsmVarArray(cFrag: var array[4, float32], v: float32) {.inline.} =
  ## asm operands must be ELEMENTS: `cFrag[0]` = first float, not the array
  asm "\"add.f32 %0, %0, %1;\" : \"+f\"(cFrag[0]) : \"f\"(v)"
  asm "\"add.f32 %0, %0, %1;\" : \"+f\"(cFrag[1]) : \"f\"(v)"

func addAsmOpenArray(cFrag: var openArray[float32], v: float32) {.inline.} =
  ## same contract for openArray: element pointer + length
  asm "\"add.f32 %0, %0, %1;\" : \"+f\"(cFrag[0]) : \"f\"(v)"

const kernelCode = cuda:
  proc varArrayAsmKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var c = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
    addAsmVarArray(c, 5.0'f32)
    output[0] = c[0]
    output[1] = c[1]
    var o = [10.0'f32, 20.0'f32, 30.0'f32, 40.0'f32]
    addAsmOpenArray(o, 1.0'f32)
    output[2] = o[0]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[3, float32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel("varArrayAsmKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  echo "  [0]=", buf[0], " [1]=", buf[1], " [2]=", buf[2]
  doAssert buf[0] == 6.0'f32   # 1 + 5
  doAssert buf[1] == 7.0'f32   # 2 + 5
  doAssert buf[2] == 11.0'f32  # 10 + 1
  echo "  OK (test_nvrtc_var_array_asm)"

when isMainModule:
  runTest()
