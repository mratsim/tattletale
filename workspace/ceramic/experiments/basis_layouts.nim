import std/macros
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime

export int_tuples, layout_algebra

type ScaledBasis*[value: static int, mode: static (int, int)]{.pure.} = object

template E*(): int = 1
template E*(a: static int): auto = ScaledBasis[1, (a, -1)]()
template E*(a, b: static int): auto = ScaledBasis[1, (a, b)]()

template scaledTuple(c: int, mode: static (int, int)): untyped =
  macro emit(c2: int, m2: static (int, int)): untyped =
    let a = m2[0]
    let b = m2[1]
    var sub: seq[NimNode] = @[]
    if b < 0:
      for i in 0 ..< a: sub.add newLit(0)
      sub.add c2
      result = nnkTupleConstr.newTree(sub)
    else:
      for i in 0 ..< b: sub.add newLit(0)
      sub.add c2
      var top: seq[NimNode] = @[]
      for i in 0 ..< a: top.add newLit(0)
      top.add nnkTupleConstr.newTree(sub)
      result = nnkTupleConstr.newTree(top)
  emit(c, mode)

macro arithAdd(a, b: tuple): untyped =
  proc isZero(n: NimNode): bool =
    n.kind in {nnkIntLit..nnkInt64Lit, nnkUIntLit..nnkUInt64Lit} and n.intVal == 0
  proc walk(x, y: NimNode, xTy, yTy: NimNode): NimNode =
    let xIsTup = xTy.isTupleTy()
    let yIsTup = yTy.isTupleTy()
    if xIsTup and isZero(x): return y
    if yIsTup and isZero(y): return x
    if not xIsTup and not yIsTup:
      return nnkInfix.newTree(bindSym"+", x, y)
    if xIsTup and yIsTup:
      let n = max(xTy.len, yTy.len)
      var elems: seq[NimNode] = @[]
      for i in 0 ..< n:
        let xi = if i < xTy.len: getTupleIndex(x, i) else: newLit(0)
        let yi = if i < yTy.len: getTupleIndex(y, i) else: newLit(0)
        let xti = if i < xTy.len: xTy[i] else: yTy[i]
        let yti = if i < yTy.len: yTy[i] else: xTy[i]
        elems.add walk(xi, yi, xti, yti)
      return nnkPar.newTree(elems)
    if not xIsTup:
      doAssert isZero(x), "zero-extension: nonzero scalar + ArithTuple"
      return y
    doAssert isZero(y), "zero-extension: nonzero scalar + ArithTuple"
    return x
  result = walk(a, b, a.getTypeInst(), b.getTypeInst())

macro arithScale(c: int, v: tuple): untyped =
  proc walk(x: NimNode, xTy: NimNode): NimNode =
    if xTy.isTupleTy():
      var elems: seq[NimNode] = @[]
      for i in 0 ..< xTy.len:
        elems.add walk(getTupleIndex(x, i), xTy[i])
      return nnkPar.newTree(elems)
    return nnkInfix.newTree(bindSym"*", c, x)
  result = walk(v, v.getTypeInst())

macro arithAddZero(a, b: typed): untyped =
  proc isZero(n: NimNode): bool =
    n.kind in {nnkIntLit..nnkInt64Lit, nnkUIntLit..nnkUInt64Lit} and n.intVal == 0
  if isZero(a):
    return b
  if isZero(b):
    return a
  result = nnkStmtListExpr.newTree(
    nnkIfStmt.newTree(
      nnkElifBranch.newTree(
        nnkInfix.newTree(bindSym"==", a, newLit(0)),
        b),
      nnkElse.newTree(
        nnkStmtList.newTree(
          nnkCall.newTree(bindSym"raiseAssert",
            newLit("scalar + ArithTuple incompatibility"))))))

func `+`*(a, b: tuple): auto = arithAdd(a, b)
func `+`*(a: int, b: tuple): auto = arithAddZero(a, b)
func `+`*(a: tuple, b: int): auto = arithAddZero(a, b)
func `*`*(c: int, v: tuple): auto = arithScale(c, v)
func `*`*(v: tuple, c: int): auto = arithScale(c, v)

func `+`*[v1: static int, m1: static (int, int), v2: static int, m2: static (int, int)](
    a: ScaledBasis[v1, m1], b: ScaledBasis[v2, m2]): auto =
  arithAdd(scaledTuple(v1, m1), scaledTuple(v2, m2))
func `+`*[v: static int, m: static (int, int)](
    a: int, b: ScaledBasis[v, m]): auto =
  arithAdd(scaledTuple(0, m), scaledTuple(v, m))
func `+`*[v: static int, m: static (int, int)](
    a: ScaledBasis[v, m], b: int): auto =
  arithAdd(scaledTuple(v, m), scaledTuple(0, m))
func `+`*[v: static int, m: static (int, int)](
    a: tuple, b: ScaledBasis[v, m]): auto =
  arithAdd(a, scaledTuple(v, m))
func `+`*[v: static int, m: static (int, int)](
    a: ScaledBasis[v, m], b: tuple): auto =
  arithAdd(scaledTuple(v, m), b)

template `*`*[v, s: static int, m: static (int, int)](
    S: Int[s], b: ScaledBasis[v, m]): auto =
  when s == 0: 0 else: ScaledBasis[s * v, m]()
template `*`*[v, s: static int, m: static (int, int)](
    b: ScaledBasis[v, m], S: Int[s]): auto =
  when s == 0: 0 else: ScaledBasis[v * s, m]()

template `*`*[v: static int, m: static (int, int)](
    S: static int, b: ScaledBasis[v, m]): auto =
  when S == 0: 0 else: ScaledBasis[S * v, m]()
template `*`*[v: static int, m: static (int, int)](
    b: ScaledBasis[v, m], S: static int): auto =
  when S == 0: 0 else: ScaledBasis[v * S, m]()

func `*`*[v: static int, m: static (int, int), I: int](
    s: I, b: ScaledBasis[v, m]): auto = scaledTuple(s * v, m)
func `*`*[v: static int, m: static (int, int), I: int](
    b: ScaledBasis[v, m], s: I): auto = scaledTuple(s * v, m)

macro make_basis_like*(profile: typed): untyped =
  if not profile.getTypeInst().isTupleTy():
    return newLit(1)
  var builder = TupleBuilderNested.new(1, dropEmpty = false)
  for event in profile.tupleStream():
    builder.onLeaves(event):
      let path = event.path
      let mode = if path.len == 1:
        nnkPar.newTree(newLit(path[0]), newLit(-1))
      else:
        nnkPar.newTree(newLit(path[0]), newLit(path[1]))
      builder.append(nnkCall.newTree(nnkBracketExpr.newTree(
        bindSym"ScaledBasis", newLit(1), mode)))
  result = builder.emit(0, emitScalarForSize1 = false)

template make_layout*[ShT: IntOrIntTuple, v: static int, m: static (int, int)](
    shapeArg: ShT; strideArg: ScaledBasis[v, m]): auto =
  Layout[typeof(makeIntTuple(shapeArg)), ScaledBasis[v, m]](
    shape: makeIntTuple(shapeArg), stride: strideArg)

template make_identity_layout*(shapeArg: IntOrIntTuple): auto =
  make_layout(shapeArg, make_basis_like(shapeArg))

func basis_repr*(x: tuple): seq[(int, seq[int])] =
  var terms: seq[(int, seq[int])] = @[]
  proc walk(v: tuple, prefix: seq[int]) =
    var i = 0
    for f in fields(v):
      when f is tuple: walk(f, prefix & @[i])
      else:
        if f != 0: terms.add ((int(f), prefix & @[i]))
      inc i
  walk(x, @[])
  if terms.len == 0: terms.add ((0, @[]))
  terms

template proj*[v: static int, m: static (int, int)](
    x: tuple, basis: ScaledBasis[v, m]): auto =
  when m[1] < 0: x[m[0]] else: x[m[0]][m[1]]

template unit*[v: static int, m: static (int, int)](
    basis: ScaledBasis[v, m]): ScaledBasis[1, m] = ScaledBasis[1, m]()
