## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#      Tile shard machinery: per-lane operand views (internal)
#
# ############################################################
#
# `shard` rebuilds an epilogue's TensorView operands as per-lane views.
# The kernel calls it. The user never writes it.

import std/macros
import workspace/crucible
import ../atoms
import ../int_tuples
import ../layouts
import ../layout_constructors
import ../tensors
import ../ptr_arithmetic
import ./tiles

# ═════════════════════════════════════════════════════════════════════════
#  The per-lane view
# ═════════════════════════════════════════════════════════════════════════

template shardShape*[R, C: static int; A: static MmaAtom](): untyped =
  ## The per-lane shape of an epilogue operand view over an R×C tile:
  ## (rowTiles, colTiles, vpt).
  (Int[R div A.mnk.m](), Int[C div A.mnk.n](),
   Int[toIntVal(A.valuesPerThread(opC))]())

template shardStrides*[R, C: static int; A: static MmaAtom](
    strideRow, strideCol: static int): untyped =
  ## The per-lane strides of an epilogue operand view over an R×C tile:
  ## (A.mnk.m·strideRow, A.mnk.n·strideCol, strideCol).
  (Int[A.mnk.m * strideRow](), Int[A.mnk.n * strideCol](),
   Int[strideCol]())

template shardView*[T; R, C: static int; A: static MmaAtom](
    buf: ptr UncheckedArray[T]; strideRow, strideCol: static int;
    origin: untyped): untyped =
  ## Per-lane gmem view of an epilogue operand: shape (rowTiles,
  ## colTiles, vpt), strides (A.mnk.m·strideRow, A.mnk.n·strideCol,
  ## strideCol), base at the tile origin plus the lane's fragment cell
  ## (fm, fn) offset. Without the per-lane offset the reads would hit
  ## each subtile's corner instead of the lane's own cells.
  let lane = int(thread_index_in_threadgroup)
  let baseOff = (uint32(origin[2]) * uint32(R)) * uint32(strideRow) +
                (uint32(origin[3]) * uint32(C)) * uint32(strideCol) +
                uint32(laneFm[A](lane)) * uint32(strideRow) +
                uint32(laneFn[A](lane)) * uint32(strideCol)
  make_view(buf +% baseOff,
            make_layout((Int[R div A.mnk.m](), Int[C div A.mnk.n](),
                         Int[toIntVal(A.valuesPerThread(opC))]()),
                        (Int[A.mnk.m * strideRow](), Int[A.mnk.n * strideCol](),
                         Int[strideCol]())))

# ═════════════════════════════════════════════════════════════════════════
#  Type introspection helpers
# ═════════════════════════════════════════════════════════════════════════

proc isTensorViewType(n: NimNode): bool =
  ## True for a type AST that is `TensorView[_, _, _]`.
  n.kind == nnkBracketExpr and n[0].eqIdent("TensorView")

proc intOf(node: NimNode): int =
  ## The static value of an `Int[V]` type AST node.
  if node.kind == nnkBracketExpr and node[0].eqIdent("Int"):
    node[1].intVal
  else:
    error("shard: expected an Int[V] type node, got " & node.treeRepr, node)

proc sameTree(a, b: NimNode): bool =
  ## Structural AST equality for the shape/stride tuple nodes (Int[V]
  ## leaves), the replacement match.
  if a.kind != b.kind: return false
  if a.kind in {nnkIdent, nnkSym}: return a.eqIdent(b)
  if a.kind == nnkIntLit: return a.intVal == b.intVal
  if a.len != b.len: return false
  for i in 0 ..< a.len:
    if not sameTree(a[i], b[i]): return false
  true

proc rebuildType(epiType, origSh, origSt, newSh, newSt: NimNode): NimNode =
  ## The sharded epilogue's type: the caller's epilogue type with the
  ## operand view's shape/strides args replaced by the per-lane ones.
  result = copyNimTree(epiType)
  if result.kind == nnkBracketExpr:
    for i in 1 ..< result.len:
      if sameTree(result[i], origSh):
        result[i] = copyNimTree(newSh)
      elif sameTree(result[i], origSt):
        result[i] = copyNimTree(newSt)

# ═════════════════════════════════════════════════════════════════════════
#  The auto-generating shard
# ═════════════════════════════════════════════════════════════════════════

macro shard*(epi: typed; buf: untyped; origin: untyped; tile: typed): untyped =
  ## The sharded epilogue value: per TensorView operand field, the
  ## per-lane gmem view over `buf`; the other fields copied from `epi`.
  ## Epilogues without TensorView captures return unchanged.
  ## The tile's static args (R, C, the atom) derive the per-lane shape
  ## and strides.
  let epiType = epi.getTypeInst()
  let epiImpl = epiType.getTypeImpl()
  let tileType = tile.getTypeInst()

  var operands: seq[NimNode]   # the IdentDefs of TensorView fields
  var scalars: seq[NimNode]    # the IdentDefs of the other fields
  for child in epiImpl[2]:
    if child.kind == nnkIdentDefs:
      if isTensorViewType(child[1]):
        operands.add child
      else:
        scalars.add child

  if operands.len == 0:
    return epi

  let R = tileType[2]
  let C = tileType[3]
  let A = tileType[4]
  let T = tileType[1]

  var shardedType = epiType
  for f in operands:
    let shNode = f[1][2]   # (Int[R], Int[C])
    let stNode = f[1][3]   # (Int[strideRow], Int[strideCol])
    let strideRow = intOf(stNode[0])
    let strideCol = intOf(stNode[1])
    let newSh = newCall(newTree(nnkBracketExpr, bindSym"shardShape", R, C, A))
    let newSt = newCall(newTree(nnkBracketExpr, bindSym"shardStrides", R, C, A),
                        newLit(strideRow), newLit(strideCol))
    shardedType = rebuildType(shardedType, shNode, stNode, newSh, newSt)

  var obj = newNimNode(nnkObjConstr)
  obj.add shardedType
  for f in scalars:
    obj.add newTree(nnkExprColonExpr, f[0], newTree(nnkDotExpr, epi, f[0]))
  for f in operands:
    let stNode = f[1][3]
    let strideRow = intOf(stNode[0])
    let strideCol = intOf(stNode[1])
    let view = newCall(newTree(nnkBracketExpr, bindSym"shardView", T, R, C, A),
                       buf, newLit(strideRow), newLit(strideCol), origin)
    obj.add newTree(nnkExprColonExpr, f[0], view)
  result = obj

# TODO: smem-staged operand copies (CUDA async-copy markers) when a backend stages epilogue operands through shared memory.
