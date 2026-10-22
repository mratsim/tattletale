## normalizeArraySpanBody deref fold contexts on the raw IR
##
## Test that normalizeArraySpanBody folds gpuDeref off index bases only,
## keeping the deref under gpuAddr and in bare element reads
##
## - gpuIndex(gpuDeref(p), i) folds to gpuIndex(p, i)
## - gpuAddr(gpuDeref(p)) survives: &(*p) is the array address (T*)
## - gpuAddr(gpuIndex(gpuDeref(p), i)) keeps the addr and folds the index
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_normalizeArraySpan.nim

import std/[tables]
import workspace/crucible/src/crucible_internal_compiler

proc runOne(node: GpuAst): GpuAst =
  ## Runs the array/span normalization pass over one fn holding `node`
  var ctx = GpuContext()
  let fn = GpuAst(kind: gpuProc, pName: newGpuIdent("anon"),
                  pRetType: GpuType(kind: gtVoid),
                  pBody: GpuAst(kind: gpuBlock, statements: @[node]),
                  pParams: @[])
  ctx.allFnTab[fn.pName] = fn
  var reg = PassRegistry.new()
  reg.registerNormalizeArraySpanParams(decayArrays = true)
  ctx.runPasses(reg)
  fn.pBody.statements[0]


proc makePtrToArray(symName: string): tuple[ident: GpuAst, deref: GpuAst] =
  ## A `p: ptr array[4, int32]` ident + its gpuDeref — the IR shape of a
  ## var-array param access BEFORE normalization (gtPtr(gtArray)).
  let int32 = GpuType(kind: gtInt32)
  let arrTyp = GpuType(kind: gtArray, aTyp: int32, aLen: 4)
  let ptrToArr = GpuType(kind: gtPtr, to: arrTyp)
  let sym = newSymbol(symName, iSym = symName & "_h", typ = ptrToArr)
  result.ident = GpuAst(kind: gpuIdent, symbol: sym)
  result.deref = GpuAst(kind: gpuDeref, dOf: result.ident)

let emptyKinds = initTable[string, seq[ArraySpanParamKind]]()

block: # 1. index context still folds: gpuIndex(gpuDeref(p), i) -> gpuIndex(p, i)
  let (_, deref) = makePtrToArray("A")
  let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: GpuType(kind: gtInt32))
  var idx = GpuAst(kind: gpuIndex, iArr: deref, iIndex: lit0)

  idx = runOne(idx)

  doAssert idx.iArr.kind == gpuIdent,
    "index-context deref must fold to the bare ident ((*p)[i] == p[i])"
  echo "  OK — index context: (*p)[i] folded to p[i]"

block: # 2. deref under gpuAddr survives: &(*p) is T*, NOT &p (T**)
  let (_, deref) = makePtrToArray("A")
  var addrNode = GpuAst(kind: gpuAddr, aOf: deref)

  addrNode = runOne(addrNode)

  doAssert addrNode.aOf.kind == gpuDeref,
    "deref under gpuAddr must NOT be stripped (&(*p) is the array address T*," &
    " stripping to &p would be the pointer variable's address T**)"
  echo "  OK — gpuAddr(gpuDeref(p)) preserved (array address T*, not T**)"

block: # 3. bare non-index deref survives (element semantics unchanged)
  let (_, deref) = makePtrToArray("A")
  var d = deref

  d = runOne(d)

  doAssert d.kind == gpuDeref,
    "non-index deref must keep its element semantics"
  echo "  OK — bare deref preserved"

block: # 4. index inside gpuAddr still folds: &p[i] (element address), deref survives only outside index
  let (_, deref) = makePtrToArray("A")
  let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: GpuType(kind: gtInt32))
  let idx = GpuAst(kind: gpuIndex, iArr: deref, iIndex: lit0)
  var addrNode = GpuAst(kind: gpuAddr, aOf: idx)

  addrNode = runOne(addrNode)

  doAssert addrNode.aOf.kind == gpuIndex,
    "gpuAddr(gpuIndex(...)) shape must survive"
  doAssert addrNode.aOf.iArr.kind == gpuIdent,
    "the index inside gpuAddr still folds: &p[i] is the element address"
  echo "  OK — gpuAddr(gpuIndex(gpuDeref(p), i)) folds inside to &p[i]"

echo "  ALL PASS — normalizeArraySpanBody deref fold is index-context-only"
