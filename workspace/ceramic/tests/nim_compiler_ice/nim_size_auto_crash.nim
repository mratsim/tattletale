## Nim ICE: lineinfos.nim(331) "cannot extract number from invalid AST node"
import std/[macros, typetraits]

type Int[V: static int] = object
type Layout[Sh, St] = object
  shape*: Sh
  stride*: St

func ceil_div(a, b: int): int =
  (a + b - 1) div b

macro genBinOp(op: untyped): untyped =
  result = newStmtList()
  result.add quote do:
    template `op`[V, U: static int](a: Int[V]; b: Int[U]): Int[`op`(V, U)] =
      Int[`op`(V, U)]()
    func `op`[V: static int](a: Int[V]; b: static int): Int[`op`(V, b)] {.inline.} =
      Int[`op`(V, b)]()
    func `op`[V: static int](a: static int; b: Int[V]): Int[`op`(a, V)] {.inline.} =
      Int[`op`(a, V)]()
    template `op`[V: static int](a: Int[V]; b: int): int =
      `op`(V, b)
    template `op`[V: static int](a: int; b: Int[V]): int =
      `op`(a, V)

genBinOp(`*`)
genBinOp(`max`)
genBinOp(`ceil_div`)

template make_layout(shapeArg: typed; strideArg: typed): auto =
  Layout[typeof(shapeArg), typeof(strideArg)](
    shape: shapeArg, stride: strideArg)

template make_layout(shapeArg: typed): auto =
  Layout[typeof(shapeArg), Int[1]](
    shape: shapeArg, stride: Int[1]())

macro flattenImpl(t: typed): untyped =
  let tNode = t
  let ttype = tNode.getTypeImpl()
  proc isLeaf(t: NimNode): bool =
    (t.kind == nnkSym and $t == "int") or
    (t.kind == nnkBracketExpr and $t[0] == "Int")
  proc collect(acc: var NimNode; e: NimNode; t: NimNode) =
    acc.add e
  result = newNimNode(nnkPar)
  collect(result, tNode, ttype)

proc flatten[T: int or Int or tuple](t: T): auto {.inline, noInit.} =
  flattenImpl(t)

proc complementScalar(
    sh, st, boundExpr: NimNode): NimNode {.compileTime.} =
  let gap = newCall(bindSym"max", newLit(1), st)
  let prd = newCall(bindSym"*", st, sh)
  let rem = newCall(bindSym"ceil_div", boundExpr, prd)
  result = newCall(bindSym"make_layout",
      newTree(nnkTupleConstr, gap, rem),
      newTree(nnkTupleConstr, newLit(1), prd))

macro complementImpl(sh, st, cosizeBound: typed): untyped =
  let boundExpr = cosizeBound
  doAssert sh.getTypeInst().kind != nnkTupleConstr
  result = complementScalar(sh, st, boundExpr)

proc complement(layout: Layout; cosizeBound: Int or int): auto =
  complementImpl(flatten(layout.shape), flatten(layout.stride), cosizeBound)

template fold(t: typed; startingAcc: typed; body: untyped): auto =
  block:
    let acc {.inject.} = startingAcc
    let it {.inject.} = t
    body

let comp = complement(make_layout(Int[4]()), Int[5]())
let sz = fold(Int[2](), Int[1](), acc * it)
echo "sz = ", $(typeof(sz))
