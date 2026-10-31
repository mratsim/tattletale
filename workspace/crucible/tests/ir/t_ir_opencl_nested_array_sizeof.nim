## OpenCL C nested-array sizeof emission
##
## Test that a sizeof over a nested array type prints the full C declaration
## form with every dimension, `sizeof(float [4][8])`
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/t_ir_opencl_nested_array_sizeof.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/opencl_lang

static:
  let f32 = tFloat(32)
  let nestedT = tArray(4, tArray(8, f32))
  let outT = tAptr(caDevice, f32)
  let kern = procD("k", @[param("out", outT)], tVoid(), @[
    varD("buf", nestedT, nil),
    callS(sym("__builtin_memcpy",
      tProcT([tPtr(caThread, nestedT), tPtr(caThread, nestedT), tUint(-1)], tVoid())),
      addrE(tPtr(caThread, nestedT), sym("buf", nestedT)),
      addrE(tPtr(caThread, nestedT), sym("buf", nestedT)),
      sizeofE(nestedT)),
  ])
  kern.isKernel = true
  let (src, err) = printOpenclModule(moduleD("probe", @[], @[kern]))
  doAssert err.len == 0, "the OpenCL printer must accept the module: " & err
  var sizeofLines = ""
  for line in src.splitLines():
    if "sizeof" in line:
      sizeofLines.add line & "\n"
  doAssert src.contains("sizeof(float [4][8])"),
    "a nested-array sizeof must print every dimension, got: " & sizeofLines

echo "[OK] t_ir_opencl_nested_array_sizeof"
