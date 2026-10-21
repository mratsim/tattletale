# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, algorithm]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/macros/replace_nodes
import workspace/ceramic/src/tensors
import ./tiles

# ############################################################
#
#            Nested loops generation over tiles
#
# ############################################################

func pop(tree: var NimNode): NimNode =
  result = tree[tree.len - 1]
  tree.del(tree.len - 1)

func strideSortedDims(shape, strides: seq[int]): seq[int] =
  for d in 0 ..< shape.len:
    if shape[d] != 1:
      result.add d
  result.sort(proc(a, b: int): int = cmp(strides[a], strides[b]))

func elemAccess(access: NimNode, filteredDims: seq[int], fullRank: int): NimNode =
  result = nnkBracketExpr.newTree(access)
  for d in 0 ..< fullRank:
    if d in filteredDims:
      result.add ident("_i" & $d)
    else:
      result.add newLit 0

func unwrapTensorType(tensor: NimNode): NimNode =
  var ty = tensor.getTypeInst()
  while ty.kind in {nnkVarTy, nnkRefTy, nnkStaticTy}:
    ty = ty[0]
  if not (ty.kind == nnkBracketExpr and ty.len == 4 and
      $ty[0] in ["TensorOwned", "Tile"]):
    error "forEach: expected a Tile (TensorOwned), got " & ty.repr, tensor
  ty

macro forEachImpl(index: untyped, values: untyped, tensors: varargs[typed],body: untyped): untyped =
  var
    aliasesStmt = newStmtList()
    elems = nnkBracket.newTree()

  # All tensors share the type through the caller's generic instantiation,
  # the first tensor's layout governs the loop nest.
  let ty = tensors[0].unwrapTensorType()
  let shape = toSeqStaticInts(ty[2])
  let strides = toSeqStaticInts(ty[3])
  let filteredDims = strideSortedDims(shape, strides)

  for i, tensor in tensors:
    # - `var` parameters are HiddenDeref
    # - Other expressions bind through evalOnceAs:
    #   - lvalues are aliased through a template
    #   - rvalue are assigned to a `let` or a `const` depending of their constness
    let ti = ident("_t" & $i)
    let accessor =
      if tensor.kind == nnkHiddenDeref:
        if tensor[0].kind in {nnkIdent, nnkSym}:
          tensor[0]
        else:
          aliasesStmt.add bindSym"evalOnceAs".newCall(ti, tensor)
          ti
      else:
        aliasesStmt.add bindSym"evalOnceAs".newCall(ti, tensor)
        ti

    elems.add elemAccess(accessor, filteredDims, shape.len)

  # Substitute the bindings with indexed access.
  var targets: seq[tuple[what: string, by: NimNode]]
  for j, val in values:
    let v = if val.kind == nnkVarTy: val[0] else: val
    targets.add ($v, elems[j])
  let newBody = body.replaceNodes(targets)

  # The optional leading index binds the flat data offset
  var indexedBody = newStmtList()
  if index.kind != nnkEmpty:
    var off: NimNode = nil
    for d in filteredDims:
      let term = ident("_i" & $d) * newLit(strides[d])
      off = if off.isNil: term
            else: off + term
    indexedBody.add quote do:
      let `index`{.inject.} = `off`
  indexedBody.add newBody

  # Wrap the modes, innermost first.
  var loop = indexedBody
  for k in countdown(filteredDims.len - 1, 0):
    let d = filteredDims[k]
    let idx = ident("_i" & $d)
    let bound = newLit shape[d]
    loop = quote do:
      for `idx` in 0 ..< `bound`:
        `loop`

  result = newStmtList()
  result.add aliasesStmt
  result.add loop

macro forEach*(args: varargs[untyped]): untyped =
  ## Iterate over same-shape tiles, element by element.
  ##
  ## Example: fused-multiply-add
  ##
  ##   forEach i, var r in result, x in a, y in b, z in c:
  ##     r = x * y + z
  ##
  ## - the optional leading `i` binds the flat data offset
  ## - `var r` for write access
  ## - tiles must be same-shape and assumed same-strides,
  ##   enforced by generic instantiation of the caller
  var params = args
  let body = params.pop()

  var index = newEmptyNode()
  var values = nnkBracket.newTree()
  var tensors = nnkArgList.newTree()

  for i, arg in params:
    if i == 0 and arg.kind == nnkIdent:
      index = arg
      continue
    case arg.kind
    of nnkInfix:
      if eqIdent(arg[0], "in"):
        values.add arg[1]
        tensors.add arg[2]
      else:
        error "forEach: unexpected operator in argument #" & $i, arg
    of nnkStmtList:
      # In generic procs symbols resolve early: `x in a` arrives
      # as a resolved `contains(a, x)` call, arguments inverted.
      if arg[0].kind == nnkCall and arg[0][0].kind == nnkOpenSymChoice and
          eqIdent(arg[0][0][0], "contains"):
        tensors.add arg[0][1]
        values.add arg[0][2]
      else:
        error "forEach: unexpected argument form #" & $i, arg
    else:
      error "forEach: unexpected argument form #" & $i, arg

  result = quote do:
    forEachImpl(`index`, `values`, `tensors`):
      `body`
