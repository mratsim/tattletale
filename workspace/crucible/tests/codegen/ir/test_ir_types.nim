## Phase 1b: Type System Fixes test
##
## Verifies:
## - hasPragma returns true when magic is the second pragma
## - GpuType.== on gtGenericInst with nil gFields does NOT crash
## - gtPtr.== distinguishes mutable from immutable
## - gtPtr.hash distinguishes mutable from immutable
## - gtInvalid == gtInvalid returns false
## - GpuProcSignature.== compares params by value (Symbol identity)
## - clone preserves mutable for gtPtr
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_types.nim

import std / [macros, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/ir/gpu_type_constructors

# ═══════════════════════════════════════════════════════════════════════
# 1. hasPragma returns true when magic is the SECOND pragma (Bug 1)
# ═══════════════════════════════════════════════════════════════════════
static:
  block:
    let node = parseStmt("""
      proc foo() {.noSideEffect, magic.} = discard
    """)[0]
    doAssert node.kind == nnkProcDef
    doAssert node.hasPragma("magic"),
      "hasPragma should return true when magic is the second pragma"
    echo "  OK — hasPragma true for second pragma"

# ═══════════════════════════════════════════════════════════════════════
# 2. hasPragma returns false when no magic pragma present
# ═══════════════════════════════════════════════════════════════════════
static:
  block:
    let node = parseStmt("""
      proc foo() {.noSideEffect, inline.} = discard
    """)[0]
    doAssert node.kind == nnkProcDef
    doAssert not node.hasPragma("magic"),
      "hasPragma should return false when magic is absent"
    echo "  OK — hasPragma false for non-magic pragmas"

# ═══════════════════════════════════════════════════════════════════════
# 3. GpuType.== on gtGenericInst with nil gFields does NOT crash (Bug 2)
# ═══════════════════════════════════════════════════════════════════════
block:
  # Both have nil gFields (default from object construction)
  let a = GpuType(kind: gtGenericInst, gName: "Test")
  let b = GpuType(kind: gtGenericInst, gName: "Test")
  doAssert a == b,
    "gtGenericInst with nil gFields/gArgs should compare equal"
  echo "  OK — gtGenericInst nil gFields does not crash"

block:
  # One has nil, other has non-nil gFields
  let a = GpuType(kind: gtGenericInst, gName: "Test",
    gFields: @[GpuTypeField(name: "x", typ: GpuType(kind: gtInt32))])
  let b = GpuType(kind: gtGenericInst, gName: "Test")
  doAssert not (a == b),
    "gtGenericInst: one nil gFields other non-nil should not be equal"
  echo "  OK — gtGenericInst nil vs non-nil gFields distinguished"

# ═══════════════════════════════════════════════════════════════════════
# 4. gtPtr.== distinguishes mutable from immutable (Bug 3)
# ═══════════════════════════════════════════════════════════════════════
block:
  let baseTyp = GpuType(kind: gtInt32)
  let a = GpuType(kind: gtPtr, to: baseTyp, implicit: false, mutable: true)
  let b = GpuType(kind: gtPtr, to: baseTyp, implicit: false, mutable: false)
  let c = GpuType(kind: gtPtr, to: baseTyp, implicit: false, mutable: true)
  doAssert not (a == b),
    "gtPtr.== should distinguish mutable from immutable"
  doAssert a == c,
    "gtPtr.== should match equal mutable values"
  echo "  OK — gtPtr.== distinguishes mutable"

# ═══════════════════════════════════════════════════════════════════════
# 5. gtPtr.hash distinguishes mutable from immutable (Bug 4)
# ═══════════════════════════════════════════════════════════════════════
block:
  let baseTyp = GpuType(kind: gtInt32)
  let a = GpuType(kind: gtPtr, to: baseTyp, implicit: false, mutable: true)
  let b = GpuType(kind: gtPtr, to: baseTyp, implicit: false, mutable: false)
  doAssert hash(a) != hash(b),
    "gtPtr.hash should distinguish mutable from immutable"
  echo "  OK — gtPtr.hash distinguishes mutable"

# ═══════════════════════════════════════════════════════════════════════
# 6. gtInvalid == gtInvalid returns false (Bug 5)
# ═══════════════════════════════════════════════════════════════════════
block:
  let a = GpuType(kind: gtInvalid)
  let b = GpuType(kind: gtInvalid)
  doAssert not (a == b),
    "gtInvalid == gtInvalid should be false (two different failures)"
  echo "  OK — gtInvalid == gtInvalid returns false"

block:
  let t = GpuType(kind: gtInvalid)
  # hash should not crash
  let h = hash(t)
  discard h
  # size should return 0
  doAssert size(t) == 0,
    "size(gtInvalid) should be 0"
  # pretty should return "Invalid"
  doAssert t.pretty() == "Invalid",
    "pretty(gtInvalid) should return 'Invalid'"
  echo "  OK — gtInvalid: hash, size, pretty work correctly"

# ═══════════════════════════════════════════════════════════════════════
# 7. GpuProcSignature.== compares params by value (Bug 6)
# ═══════════════════════════════════════════════════════════════════════
block:
  # Two idents sharing the same Symbol should have equal params
  let sym = newSymbol("x", iSym = "x_hash", typ = GpuType(kind: gtInt32), symKind = gsDeviceKernelParam)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  let ident2 = GpuAst(kind: gpuIdent, symbol: sym) # same Symbol ref

  let param1 = GpuParam(ident: ident, typ: GpuType(kind: gtInt32),
    addressSpace: asFunction, passByRef: false)
  let param2 = GpuParam(ident: ident2, typ: GpuType(kind: gtInt32),
    addressSpace: asFunction, passByRef: false)

  doAssert param1 == param2,
    "GpuParam with same Symbol ref should be equal"
  echo "  OK — GpuParam.== with same Symbol ref"

block:
  # Different Symbol refs with same name should NOT be equal
  let symA = newSymbol("x", iSym = "x_one")
  let symB = newSymbol("x", iSym = "x_two") # different Symbol, different iSym
  let identA = GpuAst(kind: gpuIdent, symbol: symA)
  let identB = GpuAst(kind: gpuIdent, symbol: symB)

  let paramA = GpuParam(ident: identA, typ: GpuType(kind: gtInt32),
    addressSpace: asFunction, passByRef: false)
  let paramB = GpuParam(ident: identB, typ: GpuType(kind: gtInt32),
    addressSpace: asFunction, passByRef: false)

  doAssert not (paramA == paramB),
    "GpuParam with different Symbol refs should not be equal"
  echo "  OK — GpuParam.== different Symbol refs"

block:
  # GpuProcSignature equality via param equality
  let sym = newSymbol("x", iSym = "x_hash")
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  let param1 = GpuParam(ident: ident, typ: GpuType(kind: gtInt32),
    addressSpace: asFunction, passByRef: false)
  let param2 = GpuParam(ident: ident, typ: GpuType(kind: gtInt32),
    addressSpace: asFunction, passByRef: false)

  let sigA = GpuProcSignature(
    params: @[param1],
    retType: GpuType(kind: gtVoid))
  let sigB = GpuProcSignature(
    params: @[param2],
    retType: GpuType(kind: gtVoid))

  doAssert sigA == sigB,
    "GpuProcSignature with equal params should be equal"

  # Different retType should not be equal
  let sigC = GpuProcSignature(
    params: @[param1],
    retType: GpuType(kind: gtInt32))
  doAssert not (sigA == sigC),
    "GpuProcSignature with different retType should not be equal"
  echo "  OK — GpuProcSignature.== compares params by value"

# ═══════════════════════════════════════════════════════════════════════
# 8. Clone preserves mutable for gtPtr (Bug 7)
# ═══════════════════════════════════════════════════════════════════════
block:
  let t = GpuType(kind: gtPtr,
    to: GpuType(kind: gtInt32),
    implicit: false,
    mutable: true)
  let c = t.clone()
  doAssert c.kind == gtPtr, "clone should preserve gtPtr kind"
  doAssert c.mutable == true,
    "clone should preserve mutable=true for gtPtr"
  echo "  OK — clone preserves mutable for gtPtr"

block:
  let t = GpuType(kind: gtPtr,
    to: GpuType(kind: gtFloat32),
    implicit: true,
    mutable: false)
  let c = t.clone()
  doAssert c.mutable == false,
    "clone should preserve mutable=false for gtPtr"
  doAssert c.implicit == true,
    "clone should preserve implicit for gtPtr"
  doAssert c.to.kind == gtFloat32,
    "clone should preserve inner type for gtPtr"
  echo "  OK — clone preserves mutable=false and other fields for gtPtr"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All type system tests passed."
