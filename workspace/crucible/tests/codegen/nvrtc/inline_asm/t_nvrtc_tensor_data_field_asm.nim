## NVRTC: `var array` and `var openArray` params lower as `T*`,
## inline-asm operands referencing elements. Both operand forms:
##   - plain `var array` local
##   - a tensor's data FIELD (`gpuAddr(gpuDot)`)
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/inline_asm/t_nvrtc_tensor_data_field_asm.nim
import workspace/crucible

type MiniTensor = object
  data: array[4, float32]

func addAsmVarArray(cFrag: var array[4, float32], v: float32) {.inline.} =
  ## asm operands must be ELEMENTS of the param (the C-decayed array):
  ## `cFrag[0]` = first float, not the whole array
  asm "\"add.f32 %0, %0, %1;\" : \"+f\"(cFrag[0]) : \"f\"(v)"
  asm "\"add.f32 %0, %0, %1;\" : \"+f\"(cFrag[1]) : \"f\"(v)"
  asm "\"add.f32 %0, %0, %1;\" : \"+f\"(cFrag[2]) : \"f\"(v)"
  asm "\"add.f32 %0, %0, %1;\" : \"+f\"(cFrag[3]) : \"f\"(v)"

func addAsmOpenArray(cFrag: var openArray[float32], v: float32) {.inline.} =
  ## same contract for openArray: element pointer + length
  asm "\"add.f32 %0, %0, %1;\" : \"+f\"(cFrag[0]) : \"f\"(v)"
  asm "\"add.f32 %0, %0, %1;\" : \"+f\"(cFrag[1]) : \"f\"(v)"

const kernelCode = cuda:
  proc tensorDataAsmKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var t = MiniTensor(data: [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    addAsmVarArray(t.data, 5.0'f32)
    output[0] = t.data[0]
    output[1] = t.data[1]
    output[2] = t.data[2]
    output[3] = t.data[3]
    var o = MiniTensor(data: [10.0'f32, 20.0'f32, 30.0'f32, 40.0'f32])
    addAsmOpenArray(o.data, 1.0'f32)
    output[4] = o.data[0]
    output[5] = o.data[1]

const varArrayKernelCode = cuda:
  proc varArrayAsmKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var c = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
    addAsmVarArray(c, 5.0'f32)
    output[0] = c[0]
    output[1] = c[1]
    var o = [10.0'f32, 20.0'f32, 30.0'f32, 40.0'f32]
    addAsmOpenArray(o, 1.0'f32)
    output[2] = o[0]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[6, float32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(float32, buf.len)
  engine.launchKernel("tensorDataAsmKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  engine.freeDevice(oDev)
  echo "  [0]=", buf[0], " [1]=", buf[1], " [2]=", buf[2], " [3]=", buf[3], " [4]=", buf[4], " [5]=", buf[5]
  doAssert buf[0] == 6.0'f32   # 1 + 5
  doAssert buf[1] == 7.0'f32   # 2 + 5
  doAssert buf[2] == 8.0'f32   # 3 + 5
  doAssert buf[3] == 9.0'f32   # 4 + 5
  doAssert buf[4] == 11.0'f32  # 10 + 1
  doAssert buf[5] == 21.0'f32  # 20 + 1
  echo "  OK (test_nvrtc_tensor_data_field_asm)"

  var buf2: array[3, float32]
  let oDev2 = engine.allocDevice(float32, buf2.len)
  engine.launchKernel("varArrayAsmKernel", oDev2, ())
  engine.copyToHostFromDevice(addr buf2[0], oDev2, buf2.len)
  engine.freeDevice(oDev2)
  echo "  [0]=", buf2[0], " [1]=", buf2[1], " [2]=", buf2[2]
  doAssert buf2[0] == 6.0'f32   # 1 + 5
  doAssert buf2[1] == 7.0'f32   # 2 + 5
  doAssert buf2[2] == 11.0'f32  # 10 + 1
  echo "  OK (test_nvrtc_var_array_asm)"

when isMainModule:
  runTest()
