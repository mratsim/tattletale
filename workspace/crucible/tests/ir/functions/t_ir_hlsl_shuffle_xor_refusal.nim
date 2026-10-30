## Hand-built CIRv2: the HLSL printer refuses the simd_shuffle_xor builtin.
##
## Test that a call whose callee has the `shkSimdShuffleXor` kind refuses
## through the printer's refusal string, while the `shkSimdShuffle` kind still
## emits the `WaveReadLaneAt` intrinsic.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_hlsl_shuffle_xor_refusal.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_norm_aliases
import workspace/crucible/src/codegen/backends/hlsl_lang

proc runKindAssignment(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerBuiltinKindAssignment()
  result = m.runPasses(reg)

static:
  let f32 = tFloat(32)

  block: # simd_shuffle_xor has no HLSL intrinsic form
    let xorSym = sym("simd_shuffle_xor", tProcT(f32, f32, tInt(32)), builtin = true)
    let kernel = procD("k", @[param("v", f32)], tVoid(), @[
      varD("r", f32, callE(xorSym, sym("v", f32), intLit(1, tInt(32)))),
      retS(),
    ])
    kernel.isKernel = true
    let m = moduleD("probe", @[], @[kernel])
    let kindErr = m.runKindAssignment()
    doAssert kindErr.len == 0, "the kind-assignment pass must succeed: " & kindErr
    let (_, err) = printHlslModule(m)
    doAssert err.contains("simd_shuffle_xor has no") and
      err.contains("lane-index xor"),
      "the HLSL printer must refuse simd_shuffle_xor, got: " & err

  block: # hazard negative: simd_shuffle still emits WaveReadLaneAt
    let shuffleSym = sym("simd_shuffle", tProcT(f32, f32, tInt(32)), builtin = true)
    let kernel = procD("k", @[param("v", f32)], tVoid(), @[
      varD("r", f32, callE(shuffleSym, sym("v", f32), intLit(3, tInt(32)))),
      retS(),
    ])
    kernel.isKernel = true
    let m = moduleD("probe", @[], @[kernel])
    let kindErr = m.runKindAssignment()
    doAssert kindErr.len == 0, "the kind-assignment pass must succeed: " & kindErr
    let (src, err) = printHlslModule(m)
    doAssert err.len == 0, "simd_shuffle must print: " & err
    doAssert "WaveReadLaneAt(" in src,
      "simd_shuffle must emit WaveReadLaneAt, got:\n" & src
