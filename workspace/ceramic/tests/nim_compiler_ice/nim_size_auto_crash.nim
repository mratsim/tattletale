## CRASH REPRODUCER — zero ceramic imports.
## lineinfos.nim(331) "cannot extract number from invalid AST node"
##
## Nim compiler bug: template matching with generic Int[V] types and
## macro-generated AST causes .intVal on non-literal node.
import std/[macros, typetraits]

# ═══════ CORE TYPES (EXACT ceramic copy) ═══════
type Int*[V: static int] = object
type IntOrIntTuple* = int | Int | tuple
type StrideOrder* = enum LayoutLeft, LayoutRight
type Layout*[Sh, St] = object
  shape*: Sh
  stride*: St

# ═══════ HELPERS ═══════
func ceil_div(a, b: int): int = (a + b - 1) div b

template rank*(t: IntOrIntTuple): static int =
  when t is int or t is Int: 1 else: tupleLen(typeof(t))
template rank*(layout: Layout): static int = rank(layout.shape)

func IntCT*(val: int): NimNode {.compileTime.} =
  newNimNode(nnkObjConstr).add(newNimNode(nnkBracketExpr).add(ident"Int", newLit(val)))

func isStaticInt*(t: NimNode): bool {.compileTime.} =
  (t.kind == nnkBracketExpr and $t[0] == "Int") or t.kind == nnkIntLit
func isStaticOne*(t: NimNode): bool {.compileTime.} =
  (t.kind == nnkBracketExpr and $t[0] == "Int" and t[1].intVal == 1) or
  (t.kind == nnkIntLit and t.intVal == 1)
func getStaticInt*(t: NimNode): int {.compileTime.} =
  if t.kind == nnkBracketExpr and $t[0] == "Int": int(t[1].intVal)
  elif t.kind == nnkIntLit: int(t.intVal)
  else: error("getStaticInt on non-static: " & t.repr)

# ═══════ OPERATORS (genBinOp — EXACT ceramic) ═══════
macro genBinOp(op: untyped): untyped =
  result = newStmtList()
  result.add quote do:
    template `op`*[V, U: static int](a: Int[V]; b: Int[U]): Int[`op`(V, U)] = Int[`op`(V, U)]()
    func `op`*[V: static int](a: Int[V]; b: static int): Int[`op`(V, b)] {.inline.} = Int[`op`(V, b)]()
    func `op`*[V: static int](a: static int; b: Int[V]): Int[`op`(a, V)] {.inline.} = Int[`op`(a, V)]()
    template `op`*[V: static int](a: Int[V]; b: int): int = `op`(V, b)
    template `op`*[V: static int](a: int; b: Int[V]): int = `op`(a, V)

genBinOp(`+`); genBinOp(`-`); genBinOp(`*`); genBinOp(`div`)
genBinOp(`mod`); genBinOp(`max`); genBinOp(`min`); genBinOp(`ceil_div`)

func `abs`*(a: int): int = abs(a)
func `abs`*[V: static int](a: Int[V]): Int[V] = Int[V]()
func `sign`*(a: int): int = (if a > 0: 1 elif a < 0: -1 else: 0)
func `sign`*[V: static int](a: Int[V]): Int[1] = Int[1]()

# ═══════ makeIntTuple (EXACT ceramic) ═══════
template makeIntTupleLeaf*(leaf: int): int = leaf
template makeIntTupleLeaf*(leaf: static int): auto = Int[leaf]()
template makeIntTupleLeaf*[V: static int](x: Int[V]): Int[V] = x

macro mapLeavesWith*(t: IntOrIntTuple; body: untyped): untyped =
  let tType = t.getTypeInst()
  if tType.kind in {nnkTupleTy, nnkTupleConstr}:
    var elems: seq[NimNode]
    if t.kind == nnkTupleConstr:
      for child in t: elems.add newCall(ident"mapLeavesWith", child, body)
    else:
      for i in 0 ..< tType.len:
        let fieldAccess = nnkBracketExpr.newTree(t, newLit(i))
        elems.add newCall(ident"mapLeavesWith", fieldAccess, body)
    result = nnkTupleConstr.newTree(elems); return
  proc replaceNodes(ast, what, by: NimNode): NimNode =
    proc inspect(node: NimNode): NimNode =
      case node.kind
      of {nnkIdent, nnkSym}:
        if node.eqIdent(what): return by
        return node
      of nnkEmpty, nnkLiterals:
        return node
      else:
        result = node.kind.newTree()
        for child in node: result.add inspect(child)
    result = inspect(ast)
  result = body.replaceNodes(ident"it", t)

template makeIntTuple*(t: IntOrIntTuple): auto =
  mixin makeIntTupleLeaf; mapLeavesWith(t, makeIntTupleLeaf(it))

# ═══════ make_layout (EXACT ceramic) ═══════
template make_layout*[ShT: IntOrIntTuple](shapeArg: ShT): auto =
  ## Single-arg: column-major stride via prefix_product
  block:
    let convShape = makeIntTuple(shapeArg)
    let strideVal = prefix_product(convShape)
    Layout[typeof(convShape), typeof(strideVal)](
      shape: convShape, stride: strideVal)

template make_layout*[ShT, StT: IntOrIntTuple](shapeArg: ShT; strideArg: StT): auto =
  Layout[typeof(makeIntTuple(shapeArg)), typeof(makeIntTuple(strideArg))](
    shape: makeIntTuple(shapeArg), stride: makeIntTuple(strideArg))

# ═══════ mode (EXACT ceramic) ═══════
template mode*(layout: Layout; idx: static int): auto =
  when layout.shape is tuple:
    make_layout(layout.shape[idx], layout.stride[idx])
  else:
    static: doAssert idx == 0; layout

# ═══════ concat (EXACT ceramic tuple+tuple) ═══════
proc concat*(a, b: tuple): auto {.inline, noInit.} =
  macro impl(): untyped =
    let aN = bindSym"a"; let bN = bindSym"b"
    let aT = aN.getTypeImpl(); let bT = bN.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    for i in 0..<aT.len: result.add newTree(nnkBracketExpr, aN, newLit(i))
    for i in 0..<bT.len: result.add newTree(nnkBracketExpr, bN, newLit(i))
  impl()

proc concat*[V: static int](a: Int[V]; b: tuple): auto {.inline, noInit.} =
  macro impl(): untyped =
    let bN = bindSym"b"; let bT = bN.getTypeImpl()
    result = newNimNode(nnkPar)
    result.add bindSym"a"
    for i in 0..<bT.len: result.add newTree(nnkBracketExpr, bN, newLit(i))
  impl()

proc concat*[V: static int](a: tuple; b: Int[V]): auto {.inline, noInit.} =
  macro impl(): untyped =
    let aN = bindSym"a"; let aT = aN.getTypeImpl()
    result = newNimNode(nnkPar)
    for i in 0..<aT.len: result.add newTree(nnkBracketExpr, aN, newLit(i))
    result.add bindSym"b"
  impl()

proc concat*[V1, V2: static int](a: Int[V1]; b: Int[V2]): auto {.inline, noInit.} =
  (a, b)

proc concat*(a, b: int): auto = (a, b)

# ═══════ flatten (EXACT ceramic) ═══════
macro flattenImpl*(t: IntOrIntTuple): untyped =
  let tNode = t; let ttype = tNode.getTypeImpl()
  proc isLeaf(t: NimNode): bool =
    (t.kind == nnkSym and $t == "int") or (t.kind == nnkBracketExpr and $t[0] == "Int")
  proc collect(acc: var NimNode; e: NimNode; t: NimNode) =
    if t.kind == nnkTupleConstr:
      for idx in 0 ..< t.len:
        let fd = t[idx]; let fa = newTree(nnkBracketExpr, e, newLit(idx))
        if isLeaf(fd): acc.add fa else: collect(acc, fa, fd)
    else: acc.add e
  result = newNimNode(nnkPar); collect(result, tNode, ttype)
proc flatten*[T: IntOrIntTuple](t: T): auto {.inline, noInit.} = flattenImpl(t)

# ═══════ evalOnceAs (EXACT ceramic) ═══════
macro evalOnceAs*(alias: untyped{nkIdent}, expression: typed): untyped =
  let aName = genSym(nskTemplate, $alias)
  result = newStmtList()
  result.add quote do:
    when `expression` is static:
      const ct_tmp {.genSym.} = `expression`
      template `aName`(): untyped = ct_tmp
    else:
      let rt_tmp {.genSym.} = `expression`
      template `aName`(): untyped = rt_tmp

# ═══════ product_each (EXACT ceramic) ═══════
macro mapModesWith*(t: tuple; body: untyped): untyped =
  let tt = getTypeInst(t); let n = tt.len
  proc subst(x: NimNode; i: int; ttup: NimNode): NimNode =
    if x.kind in {nnkIdent, nnkSym} and x.eqIdent("it"):
      result = nnkBracketExpr.newTree(ttup, newLit(i))
    else: result = x.copyNimTree()
  var items: seq[NimNode]
  for i in 0 ..< n: items.add subst(body, i, t)
  result = nnkTupleConstr.newTree(items)

template fold*[T: IntOrIntTuple](t: T; startingAcc: typed; body: untyped): auto =
  when t is int or t is Int:
    block:
      let acc {.inject.} = startingAcc
      let it {.inject.} = t
      body
  else:
    when rank(t) == 0: startingAcc
    else:
      # Inline fold_recurse: walk tuple elements
      const N = rank(t)
      let f0 =
        when t[0] is int or t[0] is Int:
          block:
            let acc{.inject.}=startingAcc
            let it{.inject.}=t[0]
            body
        else: startingAcc
      when N == 1: f0
      else:
        let f1 =
          when t[1] is int or t[1] is Int:
            block:
              let acc{.inject.}=f0
              let it{.inject.}=t[1]
              body
          else: f0
        when N == 2: f1
        else:
          let f2 =
            when t[2] is int or t[2] is Int:
              block:
                let acc{.inject.}=f1
                let it{.inject.}=t[2]
                body
            else: f1
          f2

func product_each*[T: IntOrIntTuple](t: T): auto =
  mapModesWith(t): fold(it, Int[1](), acc * it)

# ═══════ prefix_product (EXACT ceramic) ═══════
template tail_accumulator(strides, shape: IntOrIntTuple; body: untyped): auto =
  const L = rank(typeof(shape)) - 1
  when shape[L] is int or shape[L] is Int:
    block:
      let acc{.inject.}=strides[L]
      let it{.inject.}=shape[L]
      body
  else: tail_accumulator(strides[L], shape[L], body)

template prefix_scanIt_recurse*(idx: static int; t: tuple; state: typed; body: untyped): untyped =
  when t[idx] is tuple:
    let it = t[idx]; let acc = state
    let subStrides = prefix_scanIt(it, acc, body)
    const L = tupleLen(typeof(it)) - 1
    let newState =
      when it[L] is int or it[L] is Int:
        block:
          let acc{.inject.}=subStrides[L]
          let it{.inject.}=it[L]
          body
      else: tail_accumulator(subStrides[L], it[L], body)
    when idx == tupleLen(t) - 1: (subStrides,)
    else: concat((subStrides,), prefix_scanIt_recurse(idx + 1, t, newState, body))
  else:
    block:
      let it{.inject.}=t[idx]
      let acc{.inject.}=state
      let newState=body
    when idx == tupleLen(t) - 1: (acc,)
    else: concat((acc,), prefix_scanIt_recurse(idx + 1, t, newState, body))

template prefix_scanIt*(t: untyped; startingAcc: auto; body: untyped): untyped =
  when t is int or t is Int: startingAcc
  else: prefix_scanIt_recurse(0, t, startingAcc, body)

template prefix_product*(shape: IntOrIntTuple): untyped =
  prefix_scanIt(shape, Int[1](), acc * it)

# ═══════ coalesce (EXACT ceramic) ═══════
macro coalesceBackward(csShape, csStride: typed; preserveTrailing: static bool = false): untyped =
  template at(n, i: untyped): untyped = newTree(nnkBracketExpr, n, newLit(i))
  let stype = csShape.getTypeInst(); let stype2 = csStride.getTypeInst()
  if stype.kind != nnkTupleConstr and stype2.kind != nnkTupleConstr:
    if (stype.kind == nnkBracketExpr and $stype[0] == "Int" and stype[1].intVal == 1) or
       (stype.kind == nnkIntLit and stype.intVal == 1):
      result = newCall(bindSym"make_layout", IntCT(1), IntCT(0))
    else: result = newCall(bindSym"make_layout", csShape, csStride)
    return
  let N = stype.len
  var resShapes, resStrides, resSTypes, resSTypes2: seq[NimNode] = @[]
  resShapes.add csShape.at(N-1); resStrides.add csStride.at(N-1)
  resSTypes.add stype[N-1]; resSTypes2.add stype2[N-1]
  if preserveTrailing:
    let lastST = stype[N-1]
    if (lastST.kind == nnkBracketExpr and $lastST[0] == "Int" and lastST[1].intVal == 1) or
       (lastST.kind == nnkIntLit and lastST.intVal == 1):
      resShapes[0] = IntCT(low(int))
      resSTypes[0] = newNimNode(nnkBracketExpr).add(ident"Int", newLit(low(int)))
  for i in countdown(N-2, 0):
    let curST = stype[i]; let curST2 = stype2[i]
    if isStaticOne(curST): continue
    if isStaticOne(resSTypes[0]):
      resShapes[0] = csShape.at(i); resStrides[0] = csStride.at(i)
      resSTypes[0] = curST; resSTypes2[0] = curST2; continue
    if isStaticInt(curST) and isStaticInt(curST2) and
       isStaticInt(resSTypes2[0]) and isStaticInt(resSTypes[0]):
      let curProd = getStaticInt(curST) * getStaticInt(curST2)
      if curProd == getStaticInt(resSTypes2[0]):
        let mergedVal = getStaticInt(curST) * getStaticInt(resSTypes[0])
        resShapes[0] = IntCT(mergedVal); resStrides[0] = csStride.at(i)
        resSTypes[0] = newNimNode(nnkBracketExpr).add(ident"Int", newLit(mergedVal))
        resSTypes2[0] = curST2; continue
    resShapes.insert(csShape.at(i), 0); resStrides.insert(csStride.at(i), 0)
    resSTypes.insert(curST, 0); resSTypes2.insert(curST2, 0)
  if not preserveTrailing:
    while resShapes.len > 0 and isStaticOne(resSTypes[^1]):
      discard resShapes.pop(); discard resStrides.pop()
      discard resSTypes.pop(); discard resSTypes2.pop()
  if resShapes.len == 0:
    result = newCall(bindSym"make_layout", IntCT(1), newLit(0)); return
  var rShape = newNimNode(nnkTupleConstr); var rStride = newNimNode(nnkTupleConstr)
  for idx in 0 ..< resShapes.len: rShape.add resShapes[idx]; rStride.add resStrides[idx]
  if rShape.len == 1: rShape = rShape[0]; rStride = rStride[0]
  result = newCall(bindSym"make_layout", rShape, rStride)

proc coalesce*(layout: Layout): auto {.inline, noInit.} =
  coalesceBackward(flatten(layout.shape), flatten(layout.stride))
proc coalesce_preserve_trailing(layout: Layout): auto {.inline, noInit.} =
  coalesceBackward(flatten(layout.shape), flatten(layout.stride), preserveTrailing = true)

# ═══════ filter_zeros / filter_inactive (EXACT ceramic) ═══════
macro filterZerosFlat(sh, st: typed): untyped =
  let stT = st.getTypeInst(); let shT = sh.getTypeInst()
  if shT.kind != nnkTupleConstr:
    if stT.kind == nnkBracketExpr and $stT[0] == "Int" and stT[1].intVal == 0:
      result = IntCT(1)
    else: result = sh
    return
  result = newNimNode(nnkTupleConstr)
  for i in 0 ..< shT.len:
    let stN = stT[i]
    if stN.kind == nnkBracketExpr and $stN[0] == "Int" and stN[1].intVal == 0:
      result.add IntCT(1)
    else: result.add newTree(nnkBracketExpr, sh, newLit(i))

template filter_zeros*(layout: Layout): auto =
  let st = flatten(layout.stride)
  let sh = filterZerosFlat(flatten(layout.shape), st)
  make_layout(sh, st)

proc filter_inactive*(layout: Layout): auto {.inline.} = coalesce(filter_zeros(layout))

# ═══════ complement (EXACT ceramic) ═══════
proc complementScalar(sh, st, boundExpr: NimNode): NimNode {.compileTime.} =
  let stTyp = st.getTypeInst()
  if stTyp.kind == nnkBracketExpr and $stTyp[0] == "Int" and stTyp[1].intVal == 0:
    return newCall(bindSym"make_layout", boundExpr, newLit(1))
  let gap = newCall(bindSym"max", newLit(1), st)
  let prd = newCall(bindSym"*", st, sh)
  let rem = newCall(bindSym"ceil_div", boundExpr, prd)
  newCall(bindSym"coalesce",
    newCall(bindSym"make_layout",
      newTree(nnkTupleConstr, gap, rem),
      newTree(nnkTupleConstr, newLit(1), prd)))

macro complementImpl(sh, st, cosizeBound: typed): untyped =
  let boundExpr =
    if cosizeBound.getTypeInst().kind == nnkTupleConstr:
      newCall(bindSym"product_each", cosizeBound)
    else: cosizeBound
  if sh.getTypeInst().kind != nnkTupleConstr:
    complementScalar(sh, st, boundExpr)
  else: error("complementMulti NYI")

proc complement*(layout: Layout; cosizeBound: Int or int): auto =
  let f = filter_inactive(layout)
  complementImpl(flatten(f.shape), flatten(f.stride), cosizeBound)

# ═══════ compose (EXACT ceramic) ═══════
template composeImpl(
    modeIdx: static int;
    accShapes, accStrides, remainingShape, remainingStride: typed;
    lhsShapes, lhsStrides: tuple): auto =
  when modeIdx >= rank(lhsShapes) - 1:
    when remainingShape is Int and typeof(remainingShape) is Int[1] and rank(accShapes) != 0:
      make_layout(accShapes, accStrides)
    else:
      make_layout(concat(accShapes, remainingShape),
                  concat(accStrides, remainingStride * lhsStrides[modeIdx]))
  else:
    let currShape = lhsShapes[modeIdx]
    let currStride = lhsStrides[modeIdx]
    let absRemainingStride = abs(remainingStride)
    let nextShape = ceil_div(currShape, absRemainingStride)
    when nextShape is Int and typeof(nextShape) is Int[1]:
      let nextStride = ceil_div(absRemainingStride, currShape) * sign(remainingStride)
      composeImpl(modeIdx+1, accShapes, accStrides, remainingShape, nextStride,
                  lhsShapes, lhsStrides)
    elif remainingShape is Int and typeof(remainingShape) is Int[1]:
      let nextStride = ceil_div(absRemainingStride, currShape) * sign(remainingStride)
      composeImpl(modeIdx+1, accShapes, accStrides, remainingShape, nextStride,
                  lhsShapes, lhsStrides)
    else:
      let clampedShape = min(nextShape, remainingShape)
      composeImpl(modeIdx+1,
                  concat(accShapes, clampedShape),
                  concat(accStrides, remainingStride * currStride),
                  remainingShape div clampedShape,
                  ceil_div(absRemainingStride, currShape) * sign(remainingStride),
                  lhsShapes, lhsStrides)

proc buildStride(t: tuple; s: int or Int; idx: static int = 0): auto {.inline.} =
  when idx == rank(t)-1: concat(t[idx]*s, ())
  else: concat(t[idx]*s, buildStride(t, s, idx+1))

proc compose*[A, B: Layout](a: A, b: B): auto =
  when a.shape isnot tuple:
    when b.stride is tuple:
      let bSh = flatten(b.shape); let bSt = flatten(b.stride)
      make_layout(bSh, buildStride(bSt, a.stride))
    else: make_layout(flatten(b.shape), flatten(b.stride) * a.stride)
  else:
    let flatA = coalesce_preserve_trailing(a)
    when flatA.shape isnot tuple:
      make_layout(b.shape, b.stride * flatA.stride)
    else:
      composeImpl(0, (), (), b.shape, b.stride,
                  flatten(flatA.shape), flatten(flatA.stride))

# ═══════ logical_divide (EXACT ceramic) ═══════
# changing auto to int and using 1 instead of Int[1]()
# solves the ICE
proc size*(layout: Layout): auto = fold(flatten(layout.shape), Int[1](), acc * it)
# proc size*(layout: Layout): int = fold(flatten(layout.shape), 1, acc * it)

proc logical_divide_impl[A, B: Layout](layout: A; tiler: B): auto =
  let comp = complement(tiler, size(layout))
  let combined = make_layout((tiler.shape, comp.shape), (tiler.stride, comp.stride))
  compose(layout, combined)

template logical_divide*[L: Layout; V: static int](layout: L; tiler: Int[V]): auto =
  logical_divide_impl(layout, make_layout(tiler))

# ═══════ CRASH ═══════
let L = make_layout((Int[1](), Int[8](), Int[4]()), (Int[1](), Int[1](), Int[8]()))
let m1 = mode(L, 1)
let d1 = logical_divide(m1, Int[4]())
let r1shape = mode(d1, 1).shape
let sz = fold(r1shape, Int[1](), acc * it)
echo "sz = ", $(typeof(sz))
