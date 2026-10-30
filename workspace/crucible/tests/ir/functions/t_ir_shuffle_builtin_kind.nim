## Hand-built CIRv2: the shuffle builtins carry a typed kind.
##
## Test that the builtin-kind-assignment pass fills `shuffleBuiltin`
## from the canonical name table for `simd_shuffle` and `simd_shuffle_xor`.
## A plain (non-builtin) symbol stays at `shkNone`.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_shuffle_builtin_kind.nim

import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_norm_aliases

proc runKindAssignment(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerBuiltinKindAssignment()
  result = m.runPasses(reg)

static:
  let f32 = tFloat(32)
  let shuffleSym = sym("simd_shuffle", tProcT(f32, f32, tInt(32)), builtin = true)
  let shuffleXorSym = sym("simd_shuffle_xor", tProcT(f32, f32, tInt(32)), builtin = true)

  let kernel = procD("k", @[param("v", f32)], tVoid(), @[
    discardS(callE(shuffleSym, sym("v", f32), intLit(3, tInt(32)))),
    discardS(callE(shuffleXorSym, sym("v", f32), intLit(1, tInt(32)))),
    retS(),
  ])
  kernel.isKernel = true
  let m = moduleD("shuffleKind", @[], @[kernel])

  let err = m.runKindAssignment()
  doAssert err.len == 0, "the kind-assignment pass must succeed: " & err

  let call0 = m.procs[0].body[0].discardVal.callee
  doAssert call0.shuffleBuiltin == shkSimdShuffle,
    "simd_shuffle must kind-assign to shkSimdShuffle, got " & $call0.shuffleBuiltin
  let call1 = m.procs[0].body[1].discardVal.callee
  doAssert call1.shuffleBuiltin == shkSimdShuffleXor,
    "simd_shuffle_xor must kind-assign to shkSimdShuffleXor, got " & $call1.shuffleBuiltin

  # hazard negative: a non-builtin symbol sharing the name stays shkNone
  # printers must never dispatch it as a shuffle
  let shadow = sym("simd_shuffle", tProcT(f32, f32, tInt(32)))
  let shadowKernel = procD("k2", @[param("v", f32)], tVoid(), @[
    discardS(callE(shadow, sym("v", f32), intLit(3, tInt(32)))),
    retS(),
  ])
  shadowKernel.isKernel = true
  let m2 = moduleD("shadowShuffle", @[], @[shadowKernel])
  let err2 = m2.runKindAssignment()
  doAssert err2.len == 0, "the kind-assignment pass must succeed: " & err2
  let shadowCall = m2.procs[0].body[0].discardVal.callee
  doAssert shadowCall.shuffleBuiltin == shkNone,
    "a non-builtin symbol must stay shkNone, got " & $shadowCall.shuffleBuiltin

echo "  OK — shuffle builtins kind-assign, shadowing symbols stay shkNone"
