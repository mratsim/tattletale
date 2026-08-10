## NVRTC: `var array` and `var openArray` params passed a tensor's data
## FIELD (`gpuAddr(gpuDot)`) with inline-asm operands.
## Run with: nim cpp -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##   workspace/crucible/tests/codegen/nvrtc/test_nvrtc_tensor_data_field_asm.nim
##
## Regression: normalizeArraySpanBody only rewrote `gpuAddr(gpuIdent)`
## call args for var-array params — a FIELD arg (`t.data`) stayed
## `&t.data` = `float (*)[4]` and NVRTC rejected the asm operand
## ("asm operand type size(8) does not match ... constraint 'f'").
## The rewrite must also handle `gpuAddr(gpuDot)`: the bare field access
## C-decays to `T*`. Companion to test_nvrtc_var_array_asm (the ident
## case). Fails without the fix, bit-exact with it.
import workspace/crucible/src/codegen/nvrtc

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

var buf: array[6, float32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("tensorDataAsmKernel", buf, ())
echo "  [0]=", buf[0], " [1]=", buf[1], " [2]=", buf[2], " [3]=", buf[3], " [4]=", buf[4], " [5]=", buf[5]
doAssert buf[0] == 6.0'f32   # 1 + 5
doAssert buf[1] == 7.0'f32   # 2 + 5
doAssert buf[2] == 8.0'f32   # 3 + 5
doAssert buf[3] == 9.0'f32   # 4 + 5
doAssert buf[4] == 11.0'f32  # 10 + 1
doAssert buf[5] == 21.0'f32  # 20 + 1
echo "  OK (test_nvrtc_tensor_data_field_asm)"
