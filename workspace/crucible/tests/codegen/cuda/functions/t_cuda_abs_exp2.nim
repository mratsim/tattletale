## CUDA: `{.builtin.}` scalar-math names over float operands
##
## Test that the CUDA pipeline emits the device math names for `abs`
## and `exp2` and the emitted unit compiles clean under the device math
## declarations:
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/functions/t_cuda_abs_exp2.nim

import std/[os, osproc, strutils]
import workspace/crucible

const absExp2Cuda = cuda:
  proc mathKernel(output: ptr UncheckedArray[float32]) {.global.} =
    let x = output[0]
    output[1] = abs(x)
    output[2] = exp2(x)

const abs64Cuda = cuda:
  proc abs64Kernel(output: ptr UncheckedArray[float64]) {.global.} =
    let x = output[0]
    output[1] = abs(x)

# Device math declarations for the compile check, one entry per device
# name the printer must emit:
#
# - `abs(int)` is NVRTC's device `abs`, clang refuses a float-to-int
#   conversion of a bare `abs(float)` resolution under the error flags
# - `exp2(double)` is deliberately absent, a bare `exp2` over a float
#   operand fails compilation as undeclared
const deviceMathDecls = [
  "#define __global__",
  "#define __device__",
  "#define __forceinline__",
  "int abs(int n);",
  "float fabsf(float x);",
  "double fabs(double x);",
  "float exp2f(float x);",
].join("\n") & "\n"

proc compilesAsDeviceCode(src: string): bool =
  ## Syntax-checks one emitted CUDA unit under the device math declarations,
  ## true when clang accepts it under the error flags below.
  let dir = getTempDir() / "crucible_cuda_abs_exp2"
  createDir(dir)
  let path = dir / "emitted_unit.cc"
  writeFile(path, deviceMathDecls & "\n" & src)
  result = execCmd("clang -x c++ -fsyntax-only -Werror=absolute-value " &
    "-Werror=float-conversion -Werror=literal-conversion " &
    quoteShell(path)) == 0

proc runTest() =
  # pipeline acceptance: the const kernels above already ran the compiler
  # end to end. A refusal fails `nim c` itself.
  doAssert absExp2Cuda.len > 0
  doAssert abs64Cuda.len > 0
  doAssert compilesAsDeviceCode(absExp2Cuda),
    "float abs/exp2 unit must compile under the device math declarations"
  doAssert compilesAsDeviceCode(abs64Cuda),
    "double abs unit must compile under the device math declarations"
  echo "  OK - t_cuda_abs_exp2"

when isMainModule:
  runTest()
