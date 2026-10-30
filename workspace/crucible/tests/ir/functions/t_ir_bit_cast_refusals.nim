## Hand-built CIRv2: bit-reinterpret casts refuse the widths a printer cannot emit.
##
## Test that the CUDA printer refuses a bit cast of/to a 16-bit float and the GLSL
## printer refuses a bit cast of/to a 64-bit float, each through its refusal string,
## while the 32-bit pairs still emit the float-bits builtins.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_bit_cast_refusals.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/cuda_lang
import workspace/crucible/src/codegen/backends/glsl_lang

static:
  block: # CUDA: a 16-bit float source has no bit-reinterpret intrinsic
    let kernel = procD("k", @[param("h", tFloat(16))], tVoid(), @[
      varD("x", tUint(16), castE(tUint(16), sym("h", tFloat(16)))),
      retS(),
    ])
    kernel.isKernel = true
    let (_, err) = printCudaModule(moduleD("probe", @[], @[kernel]))
    doAssert err.contains("bit cast of a 16-bit float has no CUDA emission"),
      "the CUDA printer must refuse the 16-bit float bit cast, got: " & err

  block: # CUDA: a 16-bit float target has no bit-reinterpret intrinsic
    let kernel = procD("k", @[param("u", tUint(32))], tVoid(), @[
      varD("x", tFloat(16), castE(tFloat(16), sym("u", tUint(16)))),
      retS(),
    ])
    kernel.isKernel = true
    let (_, err) = printCudaModule(moduleD("probe", @[], @[kernel]))
    doAssert err.contains("bit cast to a 16-bit float has no CUDA emission"),
      "the CUDA printer must refuse the bit cast to a 16-bit float, got: " & err

  block: # CUDA hazard negative: the 32-bit float pair still emits the intrinsic
    let kernel = procD("k", @[param("f", tFloat(32))], tVoid(), @[
      varD("x", tUint(32), castE(tUint(32), sym("f", tFloat(32)))),
      retS(),
    ])
    kernel.isKernel = true
    let (src, err) = printCudaModule(moduleD("probe", @[], @[kernel]))
    doAssert err.len == 0, "the 32-bit float bit cast must print: " & err
    doAssert "__float_as_uint(" in src,
      "the 32-bit float bit cast must emit __float_as_uint, got:\n" & src

  block: # GLSL: a 64-bit float source has no floatBits builtin
    let kernel = procD("k", @[param("d", tFloat(64))], tVoid(), @[
      varD("x", tUint(64), castE(tUint(64), sym("d", tFloat(64)))),
      retS(),
    ])
    kernel.isKernel = true
    let (_, err) = printGlslModule(moduleD("probe", @[], @[kernel]))
    doAssert err.contains("bit cast of a 64-bit float has no GLSL emission"),
      "the GLSL printer must refuse the 64-bit float bit cast, got: " & err

  block: # GLSL: a 64-bit float target has no floatBits builtin
    let kernel = procD("k", @[param("u", tUint(64))], tVoid(), @[
      varD("x", tFloat(64), castE(tFloat(64), sym("u", tUint(64)))),
      retS(),
    ])
    kernel.isKernel = true
    let (_, err) = printGlslModule(moduleD("probe", @[], @[kernel]))
    doAssert err.contains("bit cast to a 64-bit float has no GLSL emission"),
      "the GLSL printer must refuse the bit cast to a 64-bit float, got: " & err

  block: # GLSL hazard negative: the 32-bit float pair still emits the builtin
    let kernel = procD("k", @[param("f", tFloat(32))], tVoid(), @[
      varD("x", tUint(32), castE(tUint(32), sym("f", tFloat(32)))),
      retS(),
    ])
    kernel.isKernel = true
    let (src, err) = printGlslModule(moduleD("probe", @[], @[kernel]))
    doAssert err.len == 0, "the 32-bit float bit cast must print: " & err
    doAssert "floatBitsToUint(" in src,
      "the 32-bit float bit cast must emit floatBitsToUint, got:\n" & src
