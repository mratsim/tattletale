## Phase 5: mangleNames pass test
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_mangleNames.nim

import std / [tables, sequtils, strutils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# 1. npClean — plain name unchanged
# ═══════════════════════════════════════════════════════════════════════════
block:
  let fnSym = newSymbol("myFunc", iSym = "myFunc_h1", symKind = gsProc)
  let fnIdent = GpuAst(kind: gpuIdent, symbol: fnSym)
  var ctx = GpuContext()
  ctx.fnTable["myFunc_h1"] = FnTableEntry(
    ident: fnIdent, body: nil, kind: {fkDefined}, namePolicy: npClean)

  mangleNamesImpl(ctx)

  doAssert fnIdent.symbol.name == "myFunc",
    "npClean should not change name, got: " & fnIdent.symbol.name
  echo "  OK — npClean: name unchanged"

# ═══════════════════════════════════════════════════════════════════════════
# 2. npHashSuffix — base58 suffix appended
# ═══════════════════════════════════════════════════════════════════════════
block:
  let fnSym = newSymbol("genFunc", iSym = "genFunc_X7yQz9", symKind = gsProc)
  let fnIdent = GpuAst(kind: gpuIdent, symbol: fnSym)
  var ctx = GpuContext()
  ctx.fnTable["genFunc_X7yQz9"] = FnTableEntry(
    ident: fnIdent, body: nil, kind: {fkGenericInst}, namePolicy: npHashSuffix)

  mangleNamesImpl(ctx)

  doAssert fnIdent.symbol.name.len > "genFunc_".len,
    "npHashSuffix should append suffix, got: " & fnIdent.symbol.name
  doAssert fnIdent.symbol.name.startsWith("genFunc_"),
    "Name should start with 'genFunc_', got: " & fnIdent.symbol.name
  # Should be genFunc_ + 7 base58 chars
  doAssert fnIdent.symbol.name.len == "genFunc_".len + 7,
    "Name should have 7-char suffix, got: " & fnIdent.symbol.name
  echo "  OK — npHashSuffix: base58 suffix appended"

# ═══════════════════════════════════════════════════════════════════════════
# 3. npPatch — operator renamed
# ═══════════════════════════════════════════════════════════════════════════
block:
  let fnSym = newSymbol("+", iSym = "+_h3", symKind = gsProc)
  let fnIdent = GpuAst(kind: gpuIdent, symbol: fnSym)
  var ctx = GpuContext()
  ctx.fnTable["+_h3"] = FnTableEntry(
    ident: fnIdent, body: nil, kind: {fkDefined}, namePolicy: npPatch)

  mangleNamesImpl(ctx)

  doAssert fnIdent.symbol.name == "add",
    "npPatch should rename '+' to 'add', got: " & fnIdent.symbol.name
  doAssert fnIdent.symbol.iSym == "add_h3",
    "iSym should also be patched, got: " & fnIdent.symbol.iSym
  echo "  OK — npPatch: + renamed to add"

# ═══════════════════════════════════════════════════════════════════════════
# 4. npPatch for all operators
# ═══════════════════════════════════════════════════════════════════════════
block:
  let ops = {"+": "add", "-": "sub", "*": "mul", "/": "div", "..": "range"}
  for (op, expected) in ops:
    let fnSym = newSymbol(op, iSym = op & "_h4", symKind = gsProc)
    let fnIdent = GpuAst(kind: gpuIdent, symbol: fnSym)
    var ctx = GpuContext()
    ctx.fnTable[op & "_h4"] = FnTableEntry(
      ident: fnIdent, body: nil, kind: {fkDefined}, namePolicy: npPatch)

    mangleNamesImpl(ctx)

    doAssert fnIdent.symbol.name == expected,
      "npPatch should rename '" & op & "' to '" & expected & "', got: " & fnIdent.symbol.name
  echo "  OK — npPatch: all operators renamed correctly"

# ═══════════════════════════════════════════════════════════════════════════
# 5. npUnassigned — automatic policy assignment
# ═══════════════════════════════════════════════════════════════════════════
block:
  # fkGenericInst should get npHashSuffix
  let genSym = newSymbol("gen", iSym = "gen_h5", symKind = gsProc)
  let genIdent = GpuAst(kind: gpuIdent, symbol: genSym)
  var ctx = GpuContext()
  ctx.fnTable["gen_h5"] = FnTableEntry(
    ident: genIdent, body: nil, kind: {fkGenericInst}, namePolicy: npUnassigned)

  mangleNamesImpl(ctx)

  doAssert genIdent.symbol.name.len > "gen_".len,
    "npUnassigned for generic inst should produce hash suffix"
  echo "  OK — npUnassigned: generic inst gets hash suffix"

echo ""
echo "  All mangleNames tests passed."
