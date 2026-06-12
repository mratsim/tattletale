## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Codegen flow (ASCII) — <compile-time> vs [runtime]:
##
##   effR = effective rank = number of dims after removing size-1 dims
##
##   <fillWithCpuImpl> ──► <buildSortedArrays>
##                                │ skip size-1 dims, reorder by stride
##                                ▼
##                           effR == 0? ──yes──► dst[0] = val  or  zeroMem (isZero)
##                                │
##                               no
##                                │
##                                ▼
##                          <genFillContiguityCode>
##                          walk innermost→outermost
##                          check st[d] == innerProd
##                                │
##                                ▼
##                        fillDimStart < effR?
##                         (contiguous suffix found?)
##                          /                    \
##                        yes                    no
##                         │                      │
##                         ▼                      ▼
##                    [isZero?]             [strided fill]
##                    /        \           elem-by-elem loops
##                  yes        no
##                   │          │
##                   ▼          ▼
##              zeroMem          counted for-loop
##              (fused suffix)   (fused suffix)
##
import std/[macros, algorithm]
import ./int_tuples
import ./layouts
import ./tensors
import system/memory


# ── Compile-time helpers (shared pattern with kernel_copy_cpu) ──


proc flattenElem(lay, field: NimNode; idx: NimNode; totalRank: int): NimNode {.compileTime.} =
  ## Generates `int(flatten(lay.field)[idx])` (multi-mode) or `int(flatten(lay.field))` (rank-1).
  if totalRank <= 1:
    result = quote do:
      int(flatten(`lay`.`field`))
  else:
    result = quote do:
      int(flatten(`lay`.`field`)[`idx`])

proc buildSortedArrays(
    R: int;
    rawSh, rawSt: seq[int];
    dstLay: NimNode
): tuple[
    shLit, stLit: NimNode;
    shapeVals, stVals: seq[int];
    effR, lastOk: int
] {.compileTime.} =
  ## Build stride-sorted arrays for fill (single tensor, no src).
  ##
  ## `effR` = effective rank = number of dims after filtering out size-1 dims.
  result.shLit = nnkBracket.newTree()
  result.stLit = nnkBracket.newTree()
  result.effR = 0
  result.lastOk = -1
  var okVals: seq[int]
  for k in 0 ..< R:
    
    let shVal = if k < rawSh.len: rawSh[k] else: DynamicSentinel
    if shVal != DynamicSentinel and shVal == 1:
      continue
    okVals.add k
    result.effR += 1
    if k < rawSh.len and rawSh[k] != DynamicSentinel:
      result.shLit.add newLit(rawSh[k])
      result.shapeVals.add rawSh[k]
    else:
      result.shLit.add flattenElem(dstLay, ident"shape", newLit(k), R)
      result.shapeVals.add DynamicSentinel
    if k < rawSt.len and rawSt[k] != DynamicSentinel:
      result.stLit.add newLit(rawSt[k])
      result.stVals.add rawSt[k]
    else:
      result.stLit.add flattenElem(dstLay, ident"stride", newLit(k), R)
      result.stVals.add DynamicSentinel
  if result.effR > 1:
    var known, unknown: seq[int]
    for d in 0 ..< result.effR:
      if result.stVals[d] == DynamicSentinel: unknown.add d
      else: known.add d
    if unknown.len > 0 and known.len > 0:
      let stVals = result.stVals
      known.sort do (a, b: int) -> int:
        cmp(stVals[b], stVals[a])
      let newOrder = unknown & known
      let oldSh = result.shLit
      let oldSt = result.stLit
      let oldShV = result.shapeVals
      let oldStV = result.stVals
      result.shLit = nnkBracket.newTree()
      result.stLit = nnkBracket.newTree()
      result.shapeVals = @[]
      result.stVals = @[]
      for d in newOrder:
        result.shLit.add oldSh[d]
        result.shapeVals.add oldShV[d]
        result.stLit.add oldSt[d]
        result.stVals.add oldStV[d]
      if known.len > 0:
        result.lastOk = okVals[^1]
      else:
        result.lastOk = -1

proc isZeroVal(val: NimNode): bool {.compileTime.} =
  ## Check if a value node is compile-time zero.
  case val.kind
  of nnkIntLit: result = val.intVal == 0
  of nnkFloatLit: result = val.floatVal == 0.0
  of nnkUIntLit: result = val.intVal == 0
  of nnkCall:  # e.g. `0.T` → `float32(0)`
    if val.len >= 1:
      result = isZeroVal(val[^1])
  of nnkConv:
    if val.len >= 1:
      result = isZeroVal(val[^1])
  else: discard

# ── Fill with contiguous suffix fusion ──

proc genFillContiguityCode(
    effR: int;
    shapeVals, stVals: seq[int];
    shSym, stSym: NimNode;
    dstData: NimNode;
    val: NimNode;
    isZero: bool
): NimNode {.compileTime.} =
  ## Generate fill code with contiguity-fused suffix.
  ## Fused suffix uses zeroMem for zero, element loop otherwise.
  var fillDimStart = effR
  var innerProd = 1
  for d in countdown(effR - 1, 0):
    if shapeVals[d] == DynamicSentinel:
      break  # dynamic shape → can't fuse at compile time
    if stVals[d] == innerProd:
      innerProd *= shapeVals[d]
      fillDimStart = d
    else:
      break

  let outerR = fillDimStart
  if fillDimStart < effR:
    # ── Contiguous suffix found: fuse into single copy ──
    var copyCount = newLit(1)
    for d in fillDimStart ..< effR:
      let cd = newLit(d)
      copyCount = quote do: `copyCount` * `shSym`[`cd`]

    if isZero:
      # ── Zero fill: use zeroMem ──
      let copySize = quote do: `copyCount` * sizeof(typeof(`dstData`[0]))
      if outerR == 0:
        # Fully contiguous: single zeroMem
        result = quote do:
          zeroMem(addr `dstData`[0], `copySize`)
      else:
        # Partially contiguous: outer loops + zeroMem per tile
        var baseDst: NimNode = newLit(0)
        let idxSyms = nnkBracket.newTree()
        for d in 0 ..< outerR:
          idxSyms.add ident("fm" & $d & "_")
          let iSym = idxSyms[d]
          let ds = newLit(d)
          baseDst = quote do: `baseDst` + `iSym` * `stSym`[`ds`]
        result = quote do:
          zeroMem(addr `dstData`[`baseDst`], `copySize`)
        for d in countdown(outerR - 1, 0):
          let iSym = idxSyms[d]
          let sd = newLit(d)
          result = quote do:
            for `iSym` in 0 ..< `shSym`[`sd`]:
              `result`
    else:
      # ── Non-zero fill path: element loop (sequential → auto-vectorized) ──
      if outerR == 0:
        # Fully contiguous: counted loop over flat range
        let N = genSym(nskLet, "N")
        let i = genSym(nskForVar, "i")
        result = quote do:
          let `N` = `copyCount`
          for `i` in 0 ..< `N`:
            `dstData`[`i`] = `val`
      else:
        # Partially contiguous: outer loops + counted inner fill
        var baseDst: NimNode = newLit(0)
        let idxSyms = nnkBracket.newTree()
        for d in 0 ..< outerR:
          idxSyms.add ident("fn" & $d & "_")
          let iSym = idxSyms[d]
          let ds = newLit(d)
          baseDst = quote do: `baseDst` + `iSym` * `stSym`[`ds`]
        let N = genSym(nskLet, "N")
        let i = genSym(nskForVar, "i")
        result = quote do:
          let `N` = `copyCount`
          for `i` in 0 ..< `N`:
            `dstData`[`baseDst` + `i`] = `val`
        for d in countdown(outerR - 1, 0):
          let iSym = idxSyms[d]
          let sd = newLit(d)
          result = quote do:
            for `iSym` in 0 ..< `shSym`[`sd`]:
              `result`
  else:
    # ── No contiguous suffix: fully strided fill element-by-element ──
    if effR == 1:
      let i0 = ident("fi0_")
      result = quote do:
        for `i0` in 0 ..< `shSym`[0]:
          `dstData`[`i0` * `stSym`[0]] = `val`
    else:
      var dstOff: NimNode = newLit(0)
      let idxSyms = nnkBracket.newTree()
      for d in 0 ..< effR:
        idxSyms.add ident("fj" & $d & "_")
        let iSym = idxSyms[d]
        let ds = newLit(d)
        dstOff = quote do:
          `dstOff` + `iSym` * `stSym`[`ds`]
      result = quote do:
        `dstData`[`dstOff`] = `val`
      for i in countdown(effR - 1, 0):
        let iSym = idxSyms[i]
        let si = newLit(i)
        result = quote do:
          for `iSym` in 0 ..< `shSym`[`si`]:
            `result`



# ── FillWith macro ──

macro fillWithCpuImpl(dst: typed; val: typed): untyped =
  let tvDst = dst.getTypeInst()
  let rawSh = toSeqStaticInts(tvDst[2])
  let rawSt = toSeqStaticInts(tvDst[3])
  # Count leaf elements in shape type
  proc flatLen(t: NimNode): int {.compileTime.} =
    if t.kind == nnkTupleConstr:
      for i in 0 ..< t.len:
        result += flatLen(t[i])
    else:
      result = 1
  let rank = flatLen(tvDst[2])

  # ── Extract data pointer and layout reference ──
  let dstData = newTree(nnkDotExpr, dst, ident"data")
  let dstLay = newTree(nnkDotExpr, dst, ident"layout")
  let isZero = isZeroVal(val)

  # ── Build stride-sorted arrays ──
  let a = buildSortedArrays(rank, rawSh, rawSt, dstLay)
  if a.effR == 0:
    # All dims are size-1 → single element fill
    if isZero:
      result = quote do:
        zeroMem(addr `dstData`[0], sizeof(typeof(`dstData`[0])))
    else:
      result = quote do:
        `dstData`[0] = `val`
    return

  # ── Bind local arrays, then generate fill loops ──
  let shSym = genSym(nskLet, "fsh")
  let stSym = genSym(nskLet, "fst")
  var stmts = newStmtList()
  stmts.add newLetStmt(shSym, a.shLit)
  stmts.add newLetStmt(stSym, a.stLit)

  stmts.add genFillContiguityCode(a.effR, a.shapeVals, a.stVals,
                                  shSym, stSym, dstData, val, isZero)
  result = stmts

# ── Public API ──

func fillWith_cpu*[T, Sh, St](tv: var TensorView[T, Sh, St]; val: T) =
  ## Fill every logical element of `tv` with `val`.
  ## Uses zeroMem for zero-fill of contiguous suffix.
  fillWithCpuImpl(tv, val)

func fillWith_cpu*[T, Sh, St](t: var Tensor[T, Sh, St]; val: T) =
  ## Fill every logical element of `t` with `val`.
  fillWithCpuImpl(t, val)
