## Adapted from Laser iter03_global_triot.nim for Ceramic tensors.
##
## nested for loops: macro-generates nested for-loops at compile-time
## with incremental base variable hoisting.
## No per-element branching — loop back-edge handles wrap.
##
## Usage: copy_laser03(dstData, dstStrides, srcData, srcStrides, shape, rank)

import std/macros


proc genNestedForLoops(rank: int;
                   shape, dstData, srcData, dstStrides, srcStrides: NimNode): NimNode =
  ## Generate nested for-loops with incremental base variable hoisting.
  ##
  ## For rank=3 produces:
  ##   for i0 in 0 ..< shape[0]:
  ##     let dstBase0 = i0 * dstStrides[0]; let srcBase0 = i0 * srcStrides[0]
  ##     for i1 in 0 ..< shape[1]:
  ##       let dstBase1 = dstBase0 + i1 * dstStrides[1]; let srcBase1 = srcBase0 + i1 * srcStrides[1]
  ##       for i2 in 0 ..< shape[2]:
  ##         dstData[dstBase1 + i2 * dstStrides[2]] = srcData[srcBase1 + i2 * srcStrides[2]]

  # ── Build innermost copy body ──
  let innerIdx = ident("i" & $(rank - 1))
  let innermostIdxLit = newLit(rank - 1)

  if rank == 1:
    result = quote do:
      for `innerIdx` in 0 ..< `shape`[0]:
        `dstData`[`innerIdx` * `dstStrides`[0]] = `srcData`[`innerIdx` * `srcStrides`[0]]
    return

  # rank >= 2: use incremental base
  let prevDst = ident("dstBase" & $(rank - 2))
  let prevSrc = ident("srcBase" & $(rank - 2))

  result = newTree(nnkForStmt, innerIdx,
    newTree(nnkInfix, ident"..<", newLit(0), newTree(nnkBracketExpr, shape, innermostIdxLit)),
    newTree(nnkStmtList,
      newTree(nnkAsgn,
        newTree(nnkBracketExpr, dstData,
          newTree(nnkInfix, ident"+", prevDst,
            newTree(nnkInfix, ident"*", innerIdx, newTree(nnkBracketExpr, dstStrides, innermostIdxLit)))),
        newTree(nnkBracketExpr, srcData,
          newTree(nnkInfix, ident"+", prevSrc,
            newTree(nnkInfix, ident"*", innerIdx, newTree(nnkBracketExpr, srcStrides, innermostIdxLit)))))))

  # Wrap in middle levels (rank-2 down to 1)
  for d in countdown(rank - 2, 1):
    let idx = ident("i" & $d)
    let baseDst = ident("dstBase" & $d)
    let baseSrc = ident("srcBase" & $d)
    let prevDst = ident("dstBase" & $(d - 1))
    let prevSrc = ident("srcBase" & $(d - 1))
    let dLit = newLit(d)
    result = newTree(nnkForStmt, idx,
      newTree(nnkInfix, ident"..<", newLit(0), newTree(nnkBracketExpr, shape, dLit)),
      newTree(nnkStmtList,
        newLetStmt(baseDst, newTree(nnkInfix, ident"+", prevDst,
          newTree(nnkInfix, ident"*", idx, newTree(nnkBracketExpr, dstStrides, dLit)))),
        newLetStmt(baseSrc, newTree(nnkInfix, ident"+", prevSrc,
          newTree(nnkInfix, ident"*", idx, newTree(nnkBracketExpr, srcStrides, dLit)))),
        result))

  # Wrap outermost level (d=0)
  if rank >= 2:
    let idx = ident("i0")
    let baseDst = ident("dstBase0")
    let baseSrc = ident("srcBase0")
    let dLit = newLit(0)
    result = newTree(nnkForStmt, idx,
      newTree(nnkInfix, ident"..<", newLit(0), newTree(nnkBracketExpr, shape, dLit)),
      newTree(nnkStmtList,
        newLetStmt(baseDst, newTree(nnkInfix, ident"*", idx, newTree(nnkBracketExpr, dstStrides, dLit))),
        newLetStmt(baseSrc, newTree(nnkInfix, ident"*", idx, newTree(nnkBracketExpr, srcStrides, dLit))),
        result))


macro genTriot(rank: static int;
               dstData, srcData, shape, dstStrides, srcStrides: untyped): untyped =
  genNestedForLoops(rank, shape, dstData, srcData, dstStrides, srcStrides)

template copy_laser03*[T; Rank: static int](
    dstData: var openArray[T]; dstStrides: array[Rank, int];
    srcData: openArray[T]; srcStrides: array[Rank, int];
    shape: array[Rank, int]) =
  ## Macro-generated TRIOT nested loops for any static rank.
  genTriot(Rank, dstData, srcData, shape, dstStrides, srcStrides)
