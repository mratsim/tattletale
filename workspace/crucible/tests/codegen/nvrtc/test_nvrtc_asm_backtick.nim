## NVRTC: inline asm with backtick identifiers — Nim's native asm symbol
## mechanism (backticked names resolve to Sym nodes).
## The gemm_mma macro generates `asm "..." : "=f"(`d0`) : "r"(`a0`)...`
## so crucible must walk the interleaved StrLit/Sym node list.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_asm_backtick.nim
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  proc asmAdd(a, b: uint32): uint32 {.device, forceinline.} =
    ## Backtick identifiers reference the proc params / locals directly.
    ## The asm body is ONE string literal (constraints inside, escaped
    ## quotes); backticked names resolve to Sym nodes.
    var res: uint32
    asm "\"add.u32 %0, %1, %2;\" : \"=r\"(`res`) : \"r\"(`a`), \"r\"(`b`)"
    return res
  proc asmMul(a, b: float32): float32 {.device, forceinline.} =
    ## The gemm_mma operand shape: "=f" output, "r" inputs.
    var res: float32
    asm "\"mul.f32 %0, %1, %2;\" : \"=f\"(`res`) : \"f\"(`a`), \"f\"(`b`)"
    return res
  proc asmBacktickKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = asmAdd(3'u32, 5'u32)
    output[1] = asmAdd(100'u32, 200'u32)
    output[2] = cast[uint32](asmMul(1.5'f32, 4.0'f32))

proc main() =
  var buf: array[3, uint32]
  var nv = initNvrtc(kernelCode)
  nv.compile()
  nv.getPtx()
  echo "PTX: ", nv.ptx.len, " bytes"
  nv.execute("asmBacktickKernel", buf, ())
  echo "  [0]=", buf[0], " [1]=", buf[1], " [2]=", buf[2]
  doAssert buf[0] == 8
  doAssert buf[1] == 300
  doAssert buf[2] == 6
  echo "  OK (test_nvrtc_asm_backtick)"

main()
