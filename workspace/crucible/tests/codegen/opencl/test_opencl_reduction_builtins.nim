## OpenCL: reduction builtin emission pin.
##
## A probe on Apple's OpenCL 1.2 runtime (cl2Metal) proved it rejects
## `sub_group_shuffle_down` in every form (bare and behind
## `cl_khr_subgroups`, the extension is ignored). OpenCL is therefore an
## emission pin: the OpenCL 2.0 core sub-group spellings are asserted as
## emitted text only.
##
## Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/test_opencl_reduction_builtins.nim

import std/strutils
import workspace/crucible

const kernelCode = opencl:
  proc reductionKernel(output: ptr UncheckedArray[float32]) {.global.} =
    let acc = output[0]
    let v = simdShuffleDown(acc, 1'u32)
    output[1] = v
    let w = simdShuffle(v, 0'u32)
    output[2] = w

proc runTest() =
  doAssert "sub_group_shuffle_down(acc, 1U)" in kernelCode,
    "OpenCL shuffleDown spelling missing:\n" & kernelCode
  doAssert "sub_group_shuffle(v, 0U)" in kernelCode,
    "OpenCL shuffle spelling missing:\n" & kernelCode
  echo "  OK — OpenCL reduction builtin emission (sub_group_shuffle_down / sub_group_shuffle)"

when isMainModule:
  runTest()
