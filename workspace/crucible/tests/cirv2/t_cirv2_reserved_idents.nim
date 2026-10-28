# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Reserved-keyword renaming: identifiers colliding with a target's reserved
## words rename with the `_nim` suffix, one common pass registered per target.
##
## Renamed sites, the target printer sees none of them:
##
## - fn names, call sites, params and their body references
## - locals (var/const) and their body references
## - struct type names, struct fields and their field accesses
## - object-construction field names
##
## Kernels below use GLSL reserved words as plain Nim identifiers:
## `shared`, `uniform`, `input`, `output` (valid Nim, invalid GLSL).
##
## Run:
##   nim c --hints:off --warnings:off --nimcache:nimcache/tests/cirv2 \
##     --outdir:build/tests/cirv2 workspace/crucible/tests/cirv2/t_cirv2_reserved_idents.nim

import std/[macros, strutils]
import
  workspace/crucible/src/builtins/builtins_pragmas,
  workspace/crucible/src/crucible_gpu_builtins,
  workspace/crucible/src/codegen/ir/cirv2_build,
  workspace/crucible/src/codegen/ir/cirv2_lang,
  workspace/crucible/src/codegen/ir/cirv2_pretty,
  workspace/crucible/src/codegen/frontend/nim_to_crucible,
  workspace/crucible/src/codegen/passes/pass_datatypes,
  workspace/crucible/src/codegen/passes/passes_legalizations,
  workspace/crucible/src/codegen/passes/passes_normalizations,
  workspace/crucible/src/codegen/passes/passes_lowering,
  workspace/crucible/src/codegen/passes/passes_low_reserved

macro traceAfterReservedPassImpl(body: typed): string =
  ## Args:
  ## - `body`: the kernel body, lowered with the GLSL reserved-word renaming
  ##   registered in the test pipeline
  ##
  ## Returns the module render as a compile-time string, refusal text becomes
  ## the compile-time error.
  let (m, err) = tryLowerCirv2(body)
  if err.len > 0:
    error(err)
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  result = newLit(renderModule(m))

macro traceAfterReservedPass*(body: untyped): string =
  ## Args:
  ## - `body`: the kernel body to lower
  ##
  ## Returns the module render after the lowering stage with the GLSL
  ## reserved-word renaming applied.
  ##
  ## Example:
  ##   const trace = traceAfterReservedPass:
  ##     proc k(o: ptr UncheckedArray[uint32]) {.global.} =
  ##       o[0] = 1'u32
  result = newCall(bindSym"traceAfterReservedPassImpl", body)

proc shared(v: float32): float32 = v * 2.0'f32

proc output(y: ptr UncheckedArray[float32], input: float32) {.global.} =
  var uniform = shared(input)
  y[0] = uniform

const reservedTrace = traceAfterReservedPass:
  output

static:
  # kernel fn name renames
  doAssert "(proc output_nim " in reservedTrace
  doAssert "(proc output " notin reservedTrace
  # param renames, its body references rename with it
  doAssert "param input_nim" in reservedTrace
  doAssert "(call shared_" in reservedTrace
  doAssert "shared_" in reservedTrace
  # local renames, its reads rename with it
  doAssert "(var uniform_nim " in reservedTrace
  doAssert "uniform_nim" in reservedTrace
  doAssert "(var uniform " notin reservedTrace
  # helper proc is an ordinary harvested proc, its instantiation-hash name
  # collides with no reserved word
  doAssert "(proc shared_" in reservedTrace

import std/[sets, tables]

static:
  # struct fields and field accesses rename through one map
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let structTy = tStruct("Pair", fld("output", f32), fld("n", f32))
  let kern = procD("kern_fields", @[param("dst", aptrD)], tVoid(), @[
    varD("p", structTy, nil),
    asgnS(dotE(sym("p", structTy), "output"), fltLit(1.0, f32)),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      dotE(sym("p", structTy), "output")),
    retS(),
  ])
  kern.isKernel = true
  let m = moduleD("fixture_fields", @[structTy], @[kern])
  var reg = PassRegistry.new()
  reg.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
  doAssert runPasses(m, reg) == ""
  # struct field renames, its accesses rename with it
  doAssert m.types[0].fields[0].name == "output_nim"
  doAssert m.types[0].fields[1].name == "n"
  var dotNames: seq[string]
  for st in kern.body:
    if st.kind == sAsgn and st.lhs.kind == eDot:
      dotNames.add st.lhs.fieldName
    if st.kind == sAsgn and st.rhs.kind == eDot:
      dotNames.add st.rhs.fieldName
  doAssert dotNames == @["output_nim", "output_nim"]

  # object-construction field names rename with the struct field they name
  let structTy2 = tStruct("Pair2", fld("layout", f32), fld("n", f32))
  let kern2 = procD("kern_oconstr", @[], tVoid(), @[
    varD("p", structTy2, oconstrE(structTy2,
      kv("layout", fltLit(2.0, f32)), kv("n", fltLit(3.0, f32)))),
    retS(),
  ])
  kern2.isKernel = true
  let m2 = moduleD("fixture_oconstr", @[structTy2], @[kern2])
  var reg2 = PassRegistry.new()
  reg2.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
  doAssert runPasses(m2, reg2) == ""
  doAssert m2.types[0].fields[0].name == "layout_nim"
  doAssert kern2.body[0].init.kvFields[0].name == "layout_nim"
  doAssert kern2.body[0].init.kvFields[1].name == "n"

  # struct type name renames, its uses inherit the renamed name
  let structTy3 = tStruct("ivec3", fld("x", f32))
  let kern3 = procD("kern_structname", @[], tVoid(), @[
    varD("v", structTy3, oconstrE(structTy3, kv("x", fltLit(1.0, f32)))),
    retS(),
  ])
  kern3.isKernel = true
  let m3 = moduleD("fixture_structname", @[structTy3], @[kern3])
  var reg3 = PassRegistry.new()
  reg3.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
  doAssert runPasses(m3, reg3) == ""
  doAssert m3.types[0].name == "ivec3_nim"
  doAssert kern3.body[0].init.ocType.name == "ivec3_nim"
