## Phase 3: FnTable + gpuFor Range Kind test
##
## Verifies:
## - Functions registered in fnTable have non-nil entries with correct kinds
## - getFnParams returns correct params from fnTable
## - FnTableEntry has correct kind flags (fkDefined, fkGenericInst, fkBuiltin)
## - gpuFor node carries fRangeKind field (rkInclusive vs rkExclusive)
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_fntable.nim

import std / [tables, sequtils, strutils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/ir/gpu_type_constructors
import workspace/crucible/src/codegen/passes/pass_datatypes
import workspace/crucible/src/codegen/gpu_compiler

# ═══════════════════════════════════════════════════════════════════════
# 1. Defined function in fnTable
# ═══════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let sym = newSymbol("myFunc", iSym = "myFunc_hash1", typ = GpuType(kind: gtVoid), symKind = gsProc)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  let body = GpuAst(kind: gpuProc, pName: ident, pRetType: GpuType(kind: gtVoid))
  ctx.fnTable["myFunc_hash1"] = FnTableEntry(
    ident: ident,
    body: body,
    kind: {fkDefined},
    namePolicy: npUnassigned
  )
  doAssert "myFunc_hash1" in ctx.fnTable, "fnTable should contain the key"
  let entry = ctx.fnTable["myFunc_hash1"]
  doAssert entry.ident != nil, "FnTableEntry should have non-nil ident"
  doAssert entry.ident.kind == gpuIdent, "Ident should be gpuIdent"
  doAssert fkDefined in entry.kind, "Entry should have fkDefined kind"
  doAssert fkBuiltin notin entry.kind, "Defined entry should NOT have fkBuiltin kind"
  doAssert entry.namePolicy == npUnassigned, "namePolicy should be npUnassigned initially"
  doAssert entry.body != nil, "Entry body should be non-nil"
  echo "  OK — Defined function in fnTable has correct fields"

# ═══════════════════════════════════════════════════════════════════════
# 2. Generic instantiation in fnTable
# ═══════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let sym = newSymbol("foo_inst", iSym = "foo_inst_hash2", typ = GpuType(kind: gtVoid), symKind = gsProc)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  let body = GpuAst(kind: gpuProc, pName: ident, pRetType: GpuType(kind: gtInt32))
  ctx.fnTable["foo_inst_hash2"] = FnTableEntry(
    ident: ident,
    body: body,
    kind: {fkGenericInst},
    namePolicy: npUnassigned
  )
  let entry = ctx.fnTable["foo_inst_hash2"]
  doAssert fkGenericInst in entry.kind, "Entry should have fkGenericInst kind"
  doAssert entry.body != nil, "Generic inst should have body"
  doAssert entry.body.kind == gpuProc, "Body should be gpuProc"
  echo "  OK — Generic instantiation in fnTable has correct fields"

# ═══════════════════════════════════════════════════════════════════════
# 3. Builtin in fnTable (body may be nil)
# ═══════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let sym = newSymbol("toOpenArray", iSym = "toOpenArray_hash3", typ = GpuType(kind: gtVoid), symKind = gsProc)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  ctx.fnTable["toOpenArray_hash3"] = FnTableEntry(
    ident: ident,
    body: nil,
    kind: {fkBuiltin},
    namePolicy: npUnassigned
  )
  let entry = ctx.fnTable["toOpenArray_hash3"]
  doAssert fkBuiltin in entry.kind, "Entry should have fkBuiltin kind"
  doAssert entry.body.isNil, "Builtin should have nil body"
  doAssert entry.namePolicy == npUnassigned, "Builtin namePolicy should be npUnassigned"
  echo "  OK — Builtin in fnTable has correct fields and nil body"

# ═══════════════════════════════════════════════════════════════════════
# 4. getFnParams returns correct params from fnTable
# ═══════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let sym = newSymbol("paramFunc", iSym = "paramFunc_hash4", typ = GpuType(kind: gtVoid), symKind = gsProc)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  var body = GpuAst(kind: gpuProc, pName: ident, pRetType: GpuType(kind: gtInt32))
  let p1 = GpuParam(
    ident: GpuAst(kind: gpuIdent, symbol: newSymbol("a", iSym = "a_h4", typ = GpuType(kind: gtInt32), symKind = gsDeviceKernelParam)),
    typ: GpuType(kind: gtInt32)
  )
  let p2 = GpuParam(
    ident: GpuAst(kind: gpuIdent, symbol: newSymbol("b", iSym = "b_h4", typ = GpuType(kind: gtFloat32), symKind = gsDeviceKernelParam)),
    typ: GpuType(kind: gtFloat32)
  )
  body.pParams = @[p1, p2]
  ctx.fnTable["paramFunc_hash4"] = FnTableEntry(
    ident: ident,
    body: body,
    kind: {fkDefined},
    namePolicy: npUnassigned
  )
  let params = ctx.getFnParams(ident)
  doAssert params.len == 2, "Should return 2 params, got " & $params.len
  doAssert params[0].typ.kind == gtInt32, "First param should be int32"
  doAssert params[1].typ.kind == gtFloat32, "Second param should be float32"
  doAssert params[0].ident.ident() == "a", "First param name should be 'a'"
  echo "  OK — getFnParams returns correct params from fnTable"

# ═══════════════════════════════════════════════════════════════════════
# 5. Multiple function kinds in single fnTable
# ═══════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  # Defined
  let dSym = newSymbol("definedFn", iSym = "definedFn_hash5", symKind = gsProc)
  let dIdent = GpuAst(kind: gpuIdent, symbol: dSym)
  ctx.fnTable["definedFn_hash5"] = FnTableEntry(
    ident: dIdent,
    body: GpuAst(kind: gpuProc, pName: dIdent, pRetType: GpuType(kind: gtVoid)),
    kind: {fkDefined},
    namePolicy: npUnassigned
  )
  # Builtin
  let bSym = newSymbol("builtinFn", iSym = "builtinFn_hash5", symKind = gsProc)
  let bIdent = GpuAst(kind: gpuIdent, symbol: bSym)
  ctx.fnTable["builtinFn_hash5"] = FnTableEntry(
    ident: bIdent,
    body: nil,
    kind: {fkBuiltin},
    namePolicy: npUnassigned
  )
  # Generic Inst
  let gSym = newSymbol("genericFn", iSym = "genericFn_hash5", symKind = gsProc)
  let gIdent = GpuAst(kind: gpuIdent, symbol: gSym)
  ctx.fnTable["genericFn_hash5"] = FnTableEntry(
    ident: gIdent,
    body: GpuAst(kind: gpuProc, pName: gIdent, pRetType: GpuType(kind: gtFloat32)),
    kind: {fkGenericInst},
    namePolicy: npUnassigned
  )
  doAssert ctx.fnTable.len == 3, "fnTable should have 3 entries"
  doAssert fkDefined in ctx.fnTable["definedFn_hash5"].kind
  doAssert fkBuiltin in ctx.fnTable["builtinFn_hash5"].kind
  doAssert fkGenericInst in ctx.fnTable["genericFn_hash5"].kind
  echo "  OK — Multiple function kinds coexist in fnTable"

# ═══════════════════════════════════════════════════════════════════════
# 6. gpuFor with rkInclusive and rkExclusive
# ═══════════════════════════════════════════════════════════════════════
block:
  let incFor = GpuAst(kind: gpuFor, fRangeKind: rkInclusive)
  let excFor = GpuAst(kind: gpuFor, fRangeKind: rkExclusive)
  doAssert incFor.fRangeKind == rkInclusive, "Inclusive for should have rkInclusive"
  doAssert excFor.fRangeKind == rkExclusive, "Exclusive for should have rkExclusive"
  doAssert incFor.fRangeKind != excFor.fRangeKind, "Range kinds should differ"
  echo "  OK — gpuFor carries rkInclusive/rkExclusive range kinds"

# ═══════════════════════════════════════════════════════════════════════
# 7. toGpuAst inclusive range (0 .. x) produces rkInclusive
# ═══════════════════════════════════════════════════════════════════════
block:
  let ir = toGpuAst:
    proc rangeTestInc(x: int32) {.device.} =
      for i in 0 .. x:
        var y = x
  doAssert ir.kind == gpuBlock
  let fn = ir.statements[0]
  doAssert fn.kind == gpuProc
  let body = fn.pBody
  var foundFor = false
  if fn.pBody.kind == gpuFor:
    foundFor = true
    doAssert fn.pBody.fRangeKind == rkInclusive, "0 .. x should produce rkInclusive, got " & $fn.pBody.fRangeKind
  else:
    for stmt in fn.pBody.statements:
      if stmt.kind == gpuFor:
        foundFor = true
        doAssert stmt.fRangeKind == rkInclusive, "0 .. x should produce rkInclusive, got " & $stmt.fRangeKind
  doAssert foundFor, "Should find a gpuFor node"
  echo "  OK — toGpuAst: 0 .. x produces rkInclusive"

# ═══════════════════════════════════════════════════════════════════════
# 8. toGpuAst exclusive range (0 ..< x) produces rkExclusive
# ═══════════════════════════════════════════════════════════════════════
block:
  let ir = toGpuAst:
    proc rangeTestExc(x: int32) {.device.} =
      for i in 0 ..< x:
        var y = x
  doAssert ir.kind == gpuBlock
  let fn = ir.statements[0]
  doAssert fn.kind == gpuProc
  # Find gpuFor in proc (body may be gpuBlock or directly gpuFor)
  var foundFor = false
  if fn.pBody.kind == gpuFor:
    foundFor = true
    doAssert fn.pBody.fRangeKind == rkExclusive, "0 ..< x should produce rkExclusive, got " & $fn.pBody.fRangeKind
  else:
    for stmt in fn.pBody.statements:
      if stmt.kind == gpuFor:
        foundFor = true
        doAssert stmt.fRangeKind == rkExclusive, "0 ..< x should produce rkExclusive, got " & $stmt.fRangeKind
  doAssert foundFor, "Should find a gpuFor node"
  echo "  OK — toGpuAst: 0 ..< x produces rkExclusive"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  FnTable variants exercised: fkDefined, fkGenericInst, fkBuiltin"
echo "  Range kind variants exercised: rkInclusive, rkExclusive"
echo "  All FnTable / range kind tests passed."
