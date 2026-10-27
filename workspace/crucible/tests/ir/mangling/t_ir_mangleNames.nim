## mangleNames pass and operator display names
##
## - plain fkDefined names keep their name, on collision every entry after
##   the first gets an iSym hash (7 base58 chars) appended
## - fkBuiltin entries emit as-is
## - operator overloads finalize at registration (getOperatorMangledName)
## - the mangleNames pass skips them
##
## mangleGpuName applies Nim's compiler/mangleutils.nim word mapping
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/mangling/t_ir_mangleNames.nim

import std / [tables, sequtils, strutils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_normalizations {.all.}
import workspace/crucible/tests/ir/ir_testutils

# ═══════════════════════════════════════════════════════════════════════
# 1. Plain name unchanged
# ═══════════════════════════════════════════════════════════════════════
static:
  let fnSym = newSymbol("myFunc", iSym = "myFunc_h1", symKind = gsProc)
  let fnIdent = GpuAst(kind: gpuIdent, symbol: fnSym)
  var ctx = GpuContext()
  ctx.fnTable["myFunc_h1"] = FnTableEntry(
    ident: fnIdent, body: nil, kind: {fkDefined})

  registerMangleNames.runPassPrepared(ctx)

  doAssert fnIdent.symbol.name == "myFunc",
    "plain name should not change, got: " & fnIdent.symbol.name
  echo "  OK — plain name unchanged"

# ═══════════════════════════════════════════════════════════════════════
# 2. Collision: the second entry gets the iSym hash
# ═══════════════════════════════════════════════════════════════════════
static:
  let symA = newSymbol("genFunc", iSym = "genFunc_AAA", symKind = gsProc)
  let identA = GpuAst(kind: gpuIdent, symbol: symA)
  let symB = newSymbol("genFunc", iSym = "genFunc_BBB", symKind = gsProc)
  let identB = GpuAst(kind: gpuIdent, symbol: symB)
  var ctx = GpuContext()
  ctx.fnTable["genFunc_AAA"] = FnTableEntry(
    ident: identA, body: nil, kind: {fkDefined})
  ctx.fnTable["genFunc_BBB"] = FnTableEntry(
    ident: identB, body: nil, kind: {fkDefined})

  registerMangleNames.runPassPrepared(ctx)

  doAssert identA.symbol.name == "genFunc",
    "first entry keeps the plain name, got: " & identA.symbol.name
  doAssert identB.symbol.name.startsWith("genFunc_"),
    "second entry gets the hash suffix, got: " & identB.symbol.name
  doAssert identB.symbol.name.len == "genFunc_".len + 7,
    "suffix is the 7-char base58 hash, got: " & identB.symbol.name
  doAssert identB.symbol.iSym == "genFunc_BBB",
    "iSym is never mangled, got: " & identB.symbol.iSym
  echo "  OK — collision: second entry gets the iSym hash"

# ═══════════════════════════════════════════════════════════════════════
# 3. fkBuiltin entries emit as-is
# ═══════════════════════════════════════════════════════════════════════
static:
  let fnSym = newSymbol("toOpenArray", iSym = "toOpenArray_h3", symKind = gsProc)
  let fnIdent = GpuAst(kind: gpuIdent, symbol: fnSym)
  var ctx = GpuContext()
  ctx.fnTable["toOpenArray_h3"] = FnTableEntry(
    ident: fnIdent, body: nil, kind: {fkBuiltin})

  registerMangleNames.runPassPrepared(ctx)

  doAssert fnIdent.symbol.name == "toOpenArray",
    "builtin name should not change, got: " & fnIdent.symbol.name
  echo "  OK — fkBuiltin emits as-is"

# ═══════════════════════════════════════════════════════════════════════
# 4. Operator-named entries are skipped (finalized at registration)
# ═══════════════════════════════════════════════════════════════════════
static:
  let fnSym = newSymbol("star_eq_1AbCdeF", iSym = "*=_h4", symKind = gsProc)
  let fnIdent = GpuAst(kind: gpuIdent, symbol: fnSym)
  var ctx = GpuContext()
  ctx.fnTable["*=_h4"] = FnTableEntry(
    ident: fnIdent, body: nil, kind: {fkDefined})

  registerMangleNames.runPassPrepared(ctx)

  doAssert fnIdent.symbol.name == "star_eq_1AbCdeF",
    "finalized operator display name must not be re-suffixed, got: " & fnIdent.symbol.name
  echo "  OK — operator-named entries skipped"

# ═══════════════════════════════════════════════════════════════════════
# 5. getOperatorMangledName — spelled-out name + iSym hash
# ═══════════════════════════════════════════════════════════════════════
static:
  for op in ["+", "-", "*", "/", "+=", "-=", "*=", "/=", "%"]:
    let display = getOperatorMangledName(op, op & "_sig")
    doAssert display.isPlainIdentifier,
      "operator display name must be a valid identifier: '" & display & "'"
    doAssert display != op,
      "the raw operator must not survive into the display name"
    doAssert getOperatorMangledName(op, op & "_sig") == display,
      "display name is deterministic in (name, iSym)"
    let other = getOperatorMangledName(op, op & "_otherSig")
    doAssert other != display,
      "distinct overloads get distinct display names: " & display & " vs " & other
  doAssert getOperatorMangledName("+", "x").startsWith("plus_"),
    "'+' spells out to plus_, got: " & getOperatorMangledName("+", "x")
  doAssert getOperatorMangledName("*=", "x").startsWith("star_eq_"),
    "'*=' spells out to star_eq_, got: " & getOperatorMangledName("*=", "x")
  doAssert getOperatorMangledName("+=", "x").startsWith("plus_eq_"),
    "'+=' spells out to plus_eq_, got: " & getOperatorMangledName("+=", "x")
  echo "  OK — getOperatorMangledName: spelled out, hashed, deterministic"

# ═══════════════════════════════════════════════════════════════════════
# 6. mangleGpuName — Nim's compiler/mangleutils.nim mapping
# ═══════════════════════════════════════════════════════════════════════
static:
  doAssert mangleGpuName("+") == "plus"
  doAssert mangleGpuName("-") == "minus"
  doAssert mangleGpuName("*") == "star"
  doAssert mangleGpuName("/") == "slash"
  doAssert mangleGpuName("=") == "eq"
  doAssert mangleGpuName("+=") == "plus_eq"
  doAssert mangleGpuName("*=") == "star_eq"
  doAssert mangleGpuName("<") == "lt"
  doAssert mangleGpuName(">") == "gt"
  doAssert mangleGpuName("..") == "dot_dot"
  doAssert mangleGpuName("$") == "dollar"
  doAssert mangleGpuName("myProc") == "myProc"
  doAssert mangleGpuName("9lives") == "X9lives"
  echo "  OK — mangleGpuName: Nim's per-character mapping"

# ═══════════════════════════════════════════════════════════════════════
# 7. mangledHashName — chars 8 positions apart must not collide
# ═══════════════════════════════════════════════════════════════════════
static:
  # the old XOR-into-byte-lanes hash mapped both to the same value
  let a = mangledHashName("foo", "foo_000000000")
  let b = mangledHashName("foo", "foo_100000001")
  doAssert a != b,
    "distinct iSyms 8 chars apart must hash differently: " & a & " vs " & b
  doAssert mangledHashName("foo", "foo_000000000") == a, "hash is deterministic"
  doAssert a.len == "foo_".len + 7, "suffix is the 7-char base58 hash"
  echo "  OK — mangledHashName: 8-apart chars no longer collide"

echo ""
echo "  All mangleNames tests passed."
