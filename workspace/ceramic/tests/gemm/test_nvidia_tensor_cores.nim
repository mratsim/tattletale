## Tests for the NVIDIA tensor-core asm layer: buildNvidiaMmaAsm (the
## backtick scalar-register format) and the gemm_mma macro expansion.
##
## The gemm_mma expansion is compile-checked in a proc that is never
## invoked: the PTX instruction would fault on CPU. Compiled = valid
## expansion.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/ceramic/tests/gemm/test_nvidia_tensor_cores.nim
import std/strutils
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernel_gemm/nvidia_tensor_cores

const mma16x8x8 = "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
  ## SM80_16x8x8_F32TF32TF32F32_TN: V_A = 4, V_B = 2, V_C = 4.

proc gemmMmaExpansion() =
  ## Compile-only: the gemm_mma expansion (scalar register locals + asm)
  ## must compile. Never invoked. The PTX would fault on CPU.
  var cFrag = make_tensor(float32, make_layout((4, 1, 1), (1, 4, 4)))
  var aFrag = make_tensor(uint32, make_layout((4, 1, 2), (1, 4, 4)))
  var bFrag = make_tensor(uint32, make_layout((2, 1, 2), (1, 2, 2)))
  gemm_mma(mma16x8x8, 4, 2, 4, cFrag, aFrag(_, _, 1), bFrag(_, _, 1))

proc main() =
  # ── buildNvidiaMmaAsm: backtick scalar-register format (gemm_mma) ──
  block:
    let s = buildNvidiaMmaAsm(mma16x8x8, 4, 2, 4,
                              "d", "a", "b", "d",
                              "float32", "uint32", "uint32", "float32")
    doAssert s.contains("{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}")
    doAssert s.contains(": \"+f\"(`d0`), \"+f\"(`d1`), \"+f\"(`d2`), \"+f\"(`d3`)")
    doAssert s.contains(": \"r\"(`a0`), \"r\"(`a1`), \"r\"(`a2`), \"r\"(`a3`)")
    doAssert s.contains(", \"r\"(`b0`), \"r\"(`b1`)")

  # ── non-aliased D/C ──
  block:
    let s = buildNvidiaMmaAsm(mma16x8x8, 4, 2, 4,
                              "d", "a", "b", "c",
                              "float32", "uint32", "uint32", "float32")
    doAssert s.contains("{%10,%11,%12,%13}")
    doAssert s.contains(": \"=f\"(`d0`)")
    doAssert s.contains(", \"f\"(`c0`), \"f\"(`c1`), \"f\"(`c2`), \"f\"(`c3`)")

  echo "  OK (test_nvidia_tensor_cores)"

main()
