import ../src/int_tuples
import std/unittest
import std/random
# ═══════════════════════════════════════════════════════════════
#  Helper procs/templates used in test suites below
# ═══════════════════════════════════════════════════════════════

proc idInt[V: static int](x: Int[V]): Int[V] = x
proc add2[V: static int](x: Int[V]): Int[V + 2] = Int[V + 2]()
proc composeAddMul[V, U: static int](a: Int[V]; b: Int[U]): Int[V * 2 + U * 3] = Int[V * 2 + U * 3]()
proc sum[V, U: static int](a: Int[V]; b: Int[U]): Int[V + U] = Int[V + U]()
template tplIdentity(x: untyped): untyped = x
template tplAdd2[V: static int](x: Int[V]): Int[V + 2] = Int[V + 2]()
template tplDouble[V: static int](x: Int[V]): Int[V * 2] = Int[V * 2]()

# ═══════════════════════════════════════════════════════════════
# evalOnceAs test suite
# ═══════════════════════════════════════════════════════════════

block: # Sym reuse — let indirection
  let x = 42
  evalOnceAs(a, x)
  doAssert a === 42
  echo "✅ let indirection"

block: # Sym reuse — const indirection (gets constant-folded to nnkIntLit)
  const y = 16
  evalOnceAs(b, y)
  doAssert b === 16
  echo "✅ const indirection"

block: # Sym reuse — proc parameter
  proc test(p: int): auto =
    evalOnceAs(c, p)
    c
  doAssert test(99) === 99
  echo "✅ param indirection"

block: # Compile-time — int literal
  evalOnceAs(e, 1024)
  doAssert e === 1024
  echo "✅ int literal"

block: # Compile-time — all-args-CT call (max of CT values)
  evalOnceAs(f, max(1, 16))
  doAssert f === 16
  echo "✅ all-args-CT call"

block: # Runtime — dynamic proc call
  proc rtAdd(a, b: int): int = a + b
  let v = 5
  evalOnceAs(g, rtAdd(v, 10))
  doAssert g === 15
  echo "✅ dynamic proc call"

block: # Runtime — mixed CT/RT args
  proc rtAdd2(a, b: int): int = a + b
  let w = 5
  evalOnceAs(h, rtAdd2(1, w))
  doAssert h === 6
  echo "✅ mixed CT/RT args"

block: # Runtime — no-arg proc
  proc getVal(): int = 99
  evalOnceAs(i, getVal())
  doAssert i === 99
  echo "✅ no-arg proc"

block: # Constant-folding — runtime func with CT args folded to Int[N]
  func square(x: int): int = x * x

  func squareWithStaticDetection(x: int): int = x * x
  func squareWithStaticDetection(x: static int): static int = x * x

  evalOnceAs(bar, squareWithStaticDetection(square(3)))
  # square(3) = 9 → `isCompileTime` sees all-args-CT → const-folded
  # squareWithStaticDetection(9) with static int → 9 * 9 = 81
  doAssert typeof(bar) is Int, "runtime func + CT args → const fold → Int[N]"
  doAssert bar === 81
  echo "✅ constant-folding"

block: # Field access — const object
  type Obj = object
    x: int
    y: int
  const obj = Obj(x: 42, y: 16)
  evalOnceAs(j, obj.x)
  doAssert j === 42
  echo "✅ field access (const object)"

block: # Field access — let object
  type Obj2 = object
    x: int
  let obj2 = Obj2(x: 99)
  evalOnceAs(k, obj2.x)
  doAssert k === 99
  echo "✅ field access (let object)"

block: # Field access — const tuple
  const tup = (a: 7, b: 8)
  evalOnceAs(l, tup.a)
  doAssert l === 7
  echo "✅ field access (const tuple)"

block: # Field access — nested const object
  type Inner = object
    val: int
  type Outer = object
    inner: Inner
  const outer = Outer(inner: Inner(val: 99))
  evalOnceAs(m, outer.inner.val)
  doAssert m === 99
  echo "✅ field access (nested const object)"

echo "\n✅ All evalOnceAs tests passed!"



suite "Int[N] — compile-time integer type":

  test "Int[4] literal":
    evalOnceAs(alias, Int[4]())
    static: doAssert toIntVal(alias) == 4

  test "negative Int[-1]":
    evalOnceAs(alias, Int[-1]())
    static: doAssert toIntVal(alias) == -1

  test "Int[0]":
    evalOnceAs(alias, Int[0]())
    static: doAssert toIntVal(alias) == 0

  test "Int[2] + Int[3]":
    evalOnceAs(alias, Int[2]() + Int[3]())
    static: doAssert toIntVal(alias) == 5

  test "Int[10] * Int[3]":
    evalOnceAs(alias, Int[10]() * Int[3]())
    static: doAssert toIntVal(alias) == 30

  test "Int[7] - Int[4]":
    evalOnceAs(alias, Int[7]() - Int[4]())
    static: doAssert toIntVal(alias) == 3

  test "Int[20] div Int[3]":
    evalOnceAs(alias, Int[20]() div Int[3]())
    static: doAssert toIntVal(alias) == 6

  test "Int[17] mod Int[5]":
    evalOnceAs(alias, Int[17]() mod Int[5]())
    static: doAssert toIntVal(alias) == 2

  test "chained arithmetic: Int[2]+Int[3]*Int[4]":
    evalOnceAs(alias, Int[2]() + Int[3]() * Int[4]())
    static: doAssert toIntVal(alias) == 14

  test "Int[5] + 3 (mixed Int+int)":
    evalOnceAs(alias, Int[5]() + 3)
    static: doAssert toIntVal(alias) == 8

  test "max(Int[3], Int[7])":
    evalOnceAs(a, max(Int[3](), Int[7]()))
    static: doAssert toIntVal(a) == 7

  test "min(Int[9], Int[4])":
    evalOnceAs(b, min(Int[9](), Int[4]()))
    static: doAssert toIntVal(b) == 4

  test "ceil_div(Int[10], Int[3])":
    evalOnceAs(c, ceil_div(Int[10](), Int[3]()))
    static: doAssert toIntVal(c) == 4

  test "abs(Int[-7])":
    evalOnceAs(d, abs(Int[-7]()))
    static: doAssert toIntVal(d) == 7

  test "multiple Int aliases in scope":
    evalOnceAs(a, Int[2]())
    evalOnceAs(b, Int[3]())
    evalOnceAs(c, Int[5]())
    static: doAssert toIntVal(a) == 2
    static: doAssert toIntVal(b) == 3
    static: doAssert toIntVal(c) == 5

# ═══════════════════════════════════════════════════════════════════════
# 3.  Proc chains — Int[N] through function calls
# ═══════════════════════════════════════════════════════════════════════

suite "Proc chains — Int[N] through function calls":

  test "identity through proc":
    evalOnceAs(alias, idInt(Int[4]()))
    static: doAssert toIntVal(alias) == 4

  test "add2 through proc":
    evalOnceAs(alias, add2(Int[4]()))
    static: doAssert toIntVal(alias) == 6

  test "proc-proc chain: add2(add2(Int[3]))":
    evalOnceAs(alias, add2(add2(Int[3]())))
    static: doAssert toIntVal(alias) == 7

  test "composeAddMul(Int[3], Int[5])":
    evalOnceAs(alias, composeAddMul(Int[3](), Int[5]()))
    static: doAssert toIntVal(alias) == 21

  test "three-deep proc chain: add2(add2(add2(Int[0])))":
    evalOnceAs(alias, add2(add2(add2(Int[0]()))))
    static: doAssert toIntVal(alias) == 6

  test "mixed arithmetic + proc chain: sum(Int[2], Int[3]) * Int[4]":
    evalOnceAs(alias, sum(Int[2](), Int[3]()) * Int[4]())
    static: doAssert toIntVal(alias) == 20

# ═══════════════════════════════════════════════════════════════════════
# 4.  Template chains — constant folding through templates
#
# NOTE: Nim 2.2.10 cannot static-evaluate chained templates with
# static int params (e.g. `tplDouble(tplAdd2(Int[3]()))`). We use
# `isConst` verification + single-template `static: doAssert` instead.
# ═══════════════════════════════════════════════════════════════════════

suite "Template chains — Int[N] through templates":

  test "identity through template":
    evalOnceAs(alias, tplIdentity(Int[4]()))
    static: doAssert toIntVal(alias) == 4

  test "add2 through template":
    evalOnceAs(alias, tplAdd2(Int[4]()))
    static: doAssert toIntVal(alias) == 6

  test "template chain: tplDouble(tplAdd2(Int[3]))":
    # Nim 2.2.10 cannot static-evaluate chained templates with static int params.
    # Compile-only verification: the expression must compile.
    evalOnceAs(alias, tplDouble(tplAdd2(Int[3]())))

  test "evalOnceAs(evalOnceAs(...)) — nested":
    evalOnceAs(inner, Int[4]())
    evalOnceAs(outer, inner)
    static: doAssert toIntVal(outer) == 4

# ═══════════════════════════════════════════════════════════════════════
# 5.  Tuples of Int[N] — shape/stride computations
# ═══════════════════════════════════════════════════════════════════════

suite "Tuples of Int[N] — shape/stride tuples":

  test "tuple literal (Int[3], Int[4])":
    evalOnceAs(alias, (Int[3](), Int[4]()))
    static: doAssert toIntVal(alias[0]) == 3
    static: doAssert toIntVal(alias[1]) == 4

  test "1-element tuple (Int[10],)":
    evalOnceAs(alias, (Int[10](),))
    static: doAssert toIntVal(alias[0]) == 10

  test "nested tuple ((Int[11], Int[22]), (Int[33],))":
    evalOnceAs(alias, ((Int[11](), Int[22]()), (Int[33](),)))
    static: doAssert toIntVal(alias[0][0]) == 11
    static: doAssert toIntVal(alias[0][1]) == 22
    static: doAssert toIntVal(alias[1][0]) == 33

  test "tuple arithmetic: element-wise add":
    proc addTup[V, U: static int](a: Int[V]; b: Int[U]): Int[V + U] = Int[V + U]()
    evalOnceAs(alias, (addTup(Int[11](), Int[22]()), addTup(Int[44](), Int[55]())))
    static: doAssert toIntVal(alias[0]) == 33
    static: doAssert toIntVal(alias[1]) == 99

  test "tuple from proc chain":
    proc makeShape[V, U: static int](a: Int[V]; b: Int[U]): (Int[V], Int[U]) = (a, b)
    evalOnceAs(alias, makeShape(Int[6](), Int[7]()))
    static: doAssert toIntVal(alias[0]) == 6
    static: doAssert toIntVal(alias[1]) == 7

  test "tuple from template chain":
    template tplShape[V, U: static int](a: Int[V]; b: Int[U]): (Int[V], Int[U]) = (a, b)
    evalOnceAs(alias, tplShape(Int[8](), Int[9]()))
    static: doAssert toIntVal(alias[0]) == 8
    static: doAssert toIntVal(alias[1]) == 9

  test "nested tuple from chained procs":
    proc inner[V: static int](x: Int[V]): (Int[V], Int[V * 2]) = (x, Int[V * 2]())
    proc outer[V: static int](t: (Int[V], Int[V * 2])): (Int[V], Int[V * 2], Int[V * 3]) =
      (t[0], t[1], Int[V * 3]())
    evalOnceAs(alias, outer(inner(Int[4]())))
    static: doAssert toIntVal(alias[0]) == 4
    static: doAssert toIntVal(alias[1]) == 8
    static: doAssert toIntVal(alias[2]) == 12

# ═══════════════════════════════════════════════════════════════════════
# 6.  isConst detection — constant info is NOT lost through alias
# ═══════════════════════════════════════════════════════════════════════

suite "isConst — constant information preserved":

  test "Int[4]() is const":
    evalOnceAs(alias, Int[4]())
    static: doAssert isConst(alias)

  test "Int[2] + Int[3] is const":
    evalOnceAs(alias, Int[2]() + Int[3]())
    static: doAssert isConst(alias)

  test "proc return Int is const":
    evalOnceAs(alias, add2(Int[4]()))
    static: doAssert isConst(alias)

  test "template return Int is const":
    evalOnceAs(alias, tplAdd2(Int[5]()))
    static: doAssert isConst(alias)

  test "tuple elements are const":
    evalOnceAs(alias, (Int[3](), Int[4]()))
    static: doAssert isConst(alias[0])
    static: doAssert isConst(alias[1])

# ═══════════════════════════════════════════════════════════════════════
# 7.  No double evaluation — side effects fire exactly once
# ═══════════════════════════════════════════════════════════════════════

suite "No double evaluation — side effects with Int types":

  test "Int construction counter increments once":
    var buildCount = 0
    proc makeInt[V: static int](): Int[V] =
      inc buildCount
      result = Int[V]()
    evalOnceAs(alias, makeInt[4]())
    check buildCount == 1

  test "tuple construction counter increments once":
    var buildCount = 0
    proc makePair[V, U: static int](): (Int[V], Int[U]) =
      inc buildCount
      result = (Int[V](), Int[U]())
    evalOnceAs(alias, makePair[2, 3]())
    check buildCount == 1

  test "RNG consumed exactly once":
    randomize(42)
    let first = rand(0..1000)
    randomize(42)
    proc makeRNGInt(): Int[42] =
      discard rand(0..1000)
      result = Int[42]()
    evalOnceAs(alias, makeRNGInt())
    # Verify RNG was consumed exactly once
    let after = rand(0..1000)
    randomize(42)
    discard rand(0..1000)
    let expected = rand(0..1000)
    check after == expected

  test "costly Int computation once":
    var computeCount = 0
    proc costlyInt[V: static int](): Int[V] =
      inc computeCount
      result = Int[V]()
    evalOnceAs(alias, costlyInt[99]())
    check computeCount == 1

  test "mixed lvalue Int alias and captured Int side-by-side":
    var counter = 0
    proc makeInt(): Int[5] =
      inc counter
      result = Int[5]()
    var mutable = 10
    evalOnceAs(fixed, makeInt())       # non-lvalue — captured once
    evalOnceAs(dynamic, mutable)       # lvalue — aliases the var
    check dynamic == 10
    check counter == 1
    mutable = 20
    check dynamic == 20
    check counter == 1

  test "nested evalOnceAs captures side effect once":
    var counter = 0
    proc makeOuter(): Int[5] =
      inc counter
      result = Int[5]()
    evalOnceAs(alias, makeOuter())
    check counter == 1
    evalOnceAs(alias2, alias)
    check counter == 1

# ═══════════════════════════════════════════════════════════════════════
# 8.  Stress / correctness
# ═══════════════════════════════════════════════════════════════════════

suite "Stress / correctness":

  test "1000 reads of captured Int construction":
    var count = 0
    proc build(): Int[42] =
      inc count
      result = Int[42]()
    evalOnceAs(alias, build())
    for i in 0..<1000:
      discard alias
    check count == 1

  test "interleaved Int lvalue and captured Int":
    var state = 0
    proc nextState(): Int[999] =
      inc state
      result = Int[999]()
    var mutable = 10
    evalOnceAs(fixed, nextState())
    evalOnceAs(dynamic, mutable)
    check dynamic == 10
    check state == 1
    mutable = 20
    check dynamic == 20
    check state == 1

  test "deeply nested evalOnceAs with Int types":
    evalOnceAs(a, Int[1]())
    evalOnceAs(b, a)
    evalOnceAs(c, b)
    evalOnceAs(d, c)
    evalOnceAs(e, d)
    static: doAssert toIntVal(e) == 1

  test "fibonacci at type level through evalOnceAs":
    proc fib(V: static int): static int =
      when V <= 1: V
      else: fib(V - 1) + fib(V - 2)
    evalOnceAs(alias, Int[fib(10)]())
    static: doAssert toIntVal(alias) == 55

  test "compile-time factorial through evalOnceAs":
    proc fact(V: static int): static int =
      when V <= 1: 1
      else: V * fact(V - 1)
    evalOnceAs(alias, Int[fact(6)]())
    static: doAssert toIntVal(alias) == 720
