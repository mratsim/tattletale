## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU GEneralized Matrix Multiply (GEMM)

import std/macros
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/macros/static_for

{.experimental: "callOperator".}
{.experimental: "dynamicBindSym".}

macro gemm_atom_at*(mma: static MmaAtom, dTile, dLoc, aTile, aLoc, bTile, bLoc: untyped): untyped =
  ## One Matrix-Multiply-Accumulate (MMA) atom call at a tile position,
  ## for an atom with 2 values per thread.
  ## Args:
  ##   - dTile, dLoc: accumulator tile and cell position
  ##   - aTile, aLoc: A operand tile and cell position
  ##   - bTile, bLoc: B operand tile and cell position
  ##
  ## This macro transforms
  ##
  ##   mma.gemm_atom_at(dTile, (m, ns), aTile, (m, k), bTile, (ns, k))
  ##
  ## into:
  ##
  ##   block gemmAtomAt:
  ##     var d0 = dTile[0, m, ns]
  ##     var d1 = dTile[1, m, ns]
  ##     let a0 = aTile[0, m, k]
  ##     let a1 = aTile[1, m, k]
  ##     let b0 = bTile[0, ns, kSlice]
  ##     let b1 = bTile[1, ns, kSlice]
  ##
  ##     mma.gemm_mma(d0, d1, a0, a1, b0, b1)
  ##
  ##     dTile[0, m, ns] = d0
  ##     dTile[1, m, ns] = d1

  let instr = constStr(mma, "instr")
  let dV = mma.valuesPerThread("cLayout")
  let aV = mma.valuesPerThread("aLayout")
  let bV = mma.valuesPerThread("bLayout")

  proc at(tile: NimNode, i: int, loc: NimNode): NimNode =
    ## tile[i, loc], an empty loc (1-D cells) indexes the value only
    if loc.kind in {nnkPar, nnkTupleConstr} and loc.len == 0:
      nnkBracketExpr.newTree(tile, newLit(i))
    else:
      nnkBracketExpr.newTree(tile, newLit(i), loc)

  result = newStmtList()

  for i in 0 ..< dV:
    result.add newVarStmt(ident("d" & $i), dTile.at(i, dLoc))
  for i in 0 ..< aV:
    result.add newLetStmt(ident("a" & $i), aTile.at(i, aLoc))
  for i in 0 ..< bV:
    result.add newLetStmt(ident("b" & $i), bTile.at(i, bLoc))

  var mmaCall = bindSym"gemm_mma".newCall(newLit(mma))

  for i in 0 ..< dV:
    mmaCall.add(ident("d" & $i))
  for i in 0 ..< aV:
    mmaCall.add(ident("a" & $i))
  for i in 0 ..< bV:
    mmaCall.add(ident("b" & $i))
  result.add mmaCall

  for i in 0 ..< dV:
    result.add newAssignment(dTile.at(i, dLoc), ident("d" & $i))

  result = nnkBlockStmt.newTree(genSym(nskLabel, "gemmAtomAt"), result)

func gemm_tile*[T, Sh, St](mma: static MmaAtom, dTile: var TensorOwned[T, Sh, St], aTile, bTile: distinct TensorOwned) =
  ## Tile-level GEneralized Matrix-Multiply (GEMM)
  ##
  ## Args:
  ##   - mma: the hardware instruction descriptor
  ##   - dTile: accumulator tile (V, MR, NR), accumulated in place
  ##   - aTile: A operand tile (V, MR, KR)
  ##   - bTile: B operand tile (V, NR, KR)
  ##
  ## Worked example, m16n8k8, warp tile (32, 16, 32): MR = NR = 2,
  ## KR = 4, one warp per tile, fp16 A/B, fp32 D.
  ##
  ## Per-lane tiles:
  ##
  ## A (V, MR, KR) = (4, 2, 4)   32 vals, 16 regs    B (V, NR, KR) = (2, 2, 4)   16 vals, 8 regs    D (V, MR, NR) = (4, 2, 2)   16 vals, 16 regs
  ##      k0   k1   k2   k3                               k0   k1   k2   k3                              n0   n1
  ## m0v0  □    □    ■    □                          n0v0  □    □    ■    □                         m0v0  ■    □
  ## m0v1  □    □    □    □                          n0v1  □    □    □    □                         m0v1  □    □
  ## m0v2  □    □    □    □                          n1v0  □    □    □    □                         m0v2  □    □
  ## m0v3  □    □    □    □                          n1v1  □    □    □    □                         m0v3  □    □
  ## m1v0  □    □    □    □                                                                         m1v0  □    □
  ## m1v1  □    □    □    □                                                                         m1v1  □    □
  ## m1v2  □    □    □    □                                                                         m1v2  □    □
  ## m1v3  □    □    □    □                                                                         m1v3  □    □
  ##
  ## warp tile: MR x NR = 2 x 2 atoms of m16n8k8
  ##        n0      n1
  ##   m0  atom00  atom01
  ##   m1  atom10  atom11
  ##
  ## one atom's D, 16x8 accumulator, cell = lane that holds it:
  ##      n0  n1  n2  n3  n4  n5  n6  n7
  ## m0   0   0   1   1   2   2   3   3
  ## m1   4   4   5   5   6   6   7   7
  ## m2   8   8   9   9   10  10  11  11
  ## m3   12  12  13  13  14  14  15  15
  ## m4   16  16  17  17  18  18  19  19
  ## m5   20  20  21  21  22  22  23  23
  ## m6   24  24  25  25  26  26  27  27
  ## m7   28  28  29  29  30  30  31  31
  ## m8   0   0   1   1   2   2   3   3
  ## m9   4   4   5   5   6   6   7   7
  ## m10  8   8   9   9   10  10  11  11
  ## m11  12  12  13  13  14  14  15  15
  ## m12  16  16  17  17  18  18  19  19
  ## m13  20  20  21  21  22  22  23  23
  ## m14  24  24  25  25  26  26  27  27
  ## m15  28  28  29  29  30  30  31  31
  ##
  ## Filled cells: one term of one D cell, the k = 2 slice.
  ##   D[0, 0, 0] = A[0, 0, 0]·B[0, 0, 0]
  ##                 + A[0, 0, 1]·B[0, 0, 1]
  ##                 + A[0, 0, 2]·B[0, 0, 2]
  ##                 + A[0, 0, 3]·B[0, 0, 3]

  template flat0(s: untyped): untyped =
    when s[0] is tuple: s[0][0] else: s[0]
  static:
    doAssert aTile.shape.flat0 === mma.valuesPerThread(opA)
    doAssert bTile.shape.flat0 === mma.valuesPerThread(opB)
    doAssert dTile.shape[0] === mma.valuesPerThread(opC)
    doAssert aTile.shape[1][1] === bTile.shape[1][1], "gemm_tile: A.KR != B.KR"
    doAssert dTile.shape[1][0] === aTile.shape[1][0], "gemm_tile: C.MR != A.MR"
    doAssert dTile.shape[1][1] === bTile.shape[1][0], "gemm_tile: C.NR != B.NR"
  const
    KR = aTile.shape[1][1].toInt()
    MR = aTile.shape[1][0].toInt()
    NR = bTile.shape[1][0].toInt()
    aBits = aTile.shape[0].toInt() * sizeof(typeof(aTile.data[0]))
    bBits = bTile.shape[0].toInt() * sizeof(typeof(bTile.data[0]))

  staticFor k, 0, KR:
    when aBits == 8:
      staticFor m, 0, MR:
        staticFor n, 0, NR:
          let ns = (when (m and 1) == 1: NR - 1 - n else: n)
          mma.gemm_atom_at(dTile, (m, ns), aTile, (m, k), bTile, (ns, k))
    elif bBits == 8:
      staticFor n, 0, NR:
        staticFor m, 0, MR:
          let ms = (when (n and 1) == 1: MR - 1 - m else: m)
          mma.gemm_atom_at(dTile, (ms, n), aTile, (ms, k), bTile, (n, k))
    else:
      staticFor n, 0, (NR + 1) div 2:
        staticFor m, 0, MR:
          let n0 = n * 2
          let ms = (when (n * 2 and 2) == 2: MR - 1 - m else: m)
          mma.gemm_atom_at(dTile, (ms, n0), aTile, (ms, k), bTile, (n0, k))
          when n * 2 + 1 < NR:
            mma.gemm_atom_at(dTile, (ms, n0 + 1), aTile, (ms, k), bTile, (n0 + 1, k))
