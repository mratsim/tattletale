# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
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

func elemAccess(access: NimNode, shape, strides: seq[int], fullRank: int): NimNode =
  var off: NimNode = nil
  for d in 0 ..< fullRank:
    if shape[d] != 1:
      let term = ident("_i" & $d) * newLit(strides[d])
      off = if off.isNil: term
            else: off + term
  let data = nnkDotExpr.newTree(access, ident("data"))
  result = nnkBracketExpr.newTree(data, if off.isNil: newLit(0) else: off)

func unwrapTensorType(tensor: NimNode): NimNode =
  var ty = tensor.getTypeInst()
  while ty.kind in {nnkVarTy, nnkRefTy, nnkStaticTy}:
    ty = ty[0]
  if not (ty.kind == nnkBracketExpr and ty.len == 4 and
      $ty[0] in ["TensorOwned", "Tile"]):
    error "forEach: expected a Tile (TensorOwned), got " & ty.repr, tensor
  ty

template forEachBody(broadcast: static bool): untyped {.dirty.} =
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

    let shapeI = toSeqStaticInts(tensor.unwrapTensorType()[2])
    let stridesI = toSeqStaticInts(tensor.unwrapTensorType()[3])
    doAssert shapeI.len == shape.len, "forEach: same rank"
    when broadcast:
      for d in 0 ..< shape.len:
        doAssert shapeI[d] == 1 or shapeI[d] == shape[d], "forEachBroadcast: mode exceeds the target's mode"
    else:
      for d in 0 ..< shape.len:
        doAssert shapeI[d] == shape[d], "forEach: same-shape only"

    elems.add elemAccess(accessor, shapeI, stridesI, shape.len)

  # Substitute the bindings with indexed access.
  var targets: seq[tuple[what: string, by: NimNode]]
  for j, val in values:
    let v = if val.kind == nnkVarTy: val[0] else: val
    targets.add ($v, elems[j])
  let newBody = body.replaceNodes(targets)

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

macro forEachImpl(index: untyped, values: untyped, tensors: varargs[typed], body: untyped): untyped =
  forEachBody(false)

macro forEachBroadcastImpl(index: untyped, values: untyped, tensors: varargs[typed], body: untyped): untyped =
  forEachBody(true)

template forEachAll(broadcast: static bool): untyped {.dirty.} =
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

  when broadcast:
    result = quote do:
      forEachBroadcastImpl(`index`, `values`, `tensors`):
        `body`
  else:
    result = quote do:
      forEachImpl(`index`, `values`, `tensors`):
        `body`

macro forEach*(args: varargs[untyped]): untyped =
  ## Iterate over same-shape tiles, element by element.
  ## A smaller operand fails to compile: broadcasting is forEachBroadcast.
  ##
  ## Example: fused-multiply-add
  ##
  ##   forEach i, var r in result, x in a, y in b, z in c:
  ##     r = x * y + z
  ##
  ## - you can add a loop iteration variable
  ## - `var r` for write access
  ## - tiles must be same-shape and assumed same-strides,
  ##   the generic instantiation enforces the contract
  forEachAll(false)

macro forEachBroadcast*(args: varargs[untyped]): untyped =
  ## Example: broadcast a (M, 1) column over an (M, N) tile
  ##
  ##   forEachBroadcast var r in result, x in a, y in col:
  ##     r = x + y
  ##
  ## - you can add a loop iteration variable
  ## - `var r` for write access
  forEachAll(true)
