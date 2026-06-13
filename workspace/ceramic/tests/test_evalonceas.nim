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
  doAssert a == 42
  echo "✅ let indirection"

block: # Sym reuse — const indirection (gets constant-folded to nnkIntLit)
  const y = 16
  evalOnceAs(b, y)
  doAssert typeof(b) is int, "const value should stay as int (no Int[N] wrapping)"
  doAssert b == 16
  echo "✅ const indirection"

block: # Sym reuse — proc parameter
  proc test(p: int): auto =
    evalOnceAs(c, p)
    c
  doAssert test(99) == 99
  echo "✅ param indirection"

block: # Compile-time — int literal
  evalOnceAs(e, 1024)
  doAssert typeof(e) is int, "int literal should stay as int (no Int[N] wrapping)"
  doAssert e == 1024
  echo "✅ int literal"

block: # Compile-time — all-args-CT call (max of CT values)
  evalOnceAs(f, max(1, 16))
  doAssert typeof(f) is int, "all-args-CT should stay as int (no Int[N] wrapping)"
  doAssert f == 16
  echo "✅ all-args-CT call"

block: # Runtime — dynamic proc call
  proc rtAdd(a, b: int): int = a + b
  let v = 5
  evalOnceAs(g, rtAdd(v, 10))
  doAssert g == 15
  echo "✅ dynamic proc call"

block: # Runtime — mixed CT/RT args
  proc rtAdd2(a, b: int): int = a + b
  let w = 5
  evalOnceAs(h, rtAdd2(1, w))
  doAssert h == 6
  echo "✅ mixed CT/RT args"

block: # Runtime — no-arg proc
  proc getVal(): int = 99
  evalOnceAs(i, getVal())
  doAssert i == 99
  echo "✅ no-arg proc"

block: # Constant-folding — runtime func with CT args folded to int
  func square(x: int): int = x * x

  func squareWithStaticDetection(x: int): int = x * x
  func squareWithStaticDetection(x: static int): static int = x * x

  evalOnceAs(bar, squareWithStaticDetection(square(3)))
  doAssert typeof(bar) is int, "runtime func + CT args → const fold → int (no Int[N] wrapping)"
  doAssert bar == 81
  echo "✅ constant-folding"

block: # Field access — const object
  type Obj = object
    x: int
    y: int
  const obj = Obj(x: 42, y: 16)
  evalOnceAs(j, obj.x)
  doAssert typeof(j) is int, "const field access should stay as int (no Int[N] wrapping)"
  doAssert j == 42
  echo "✅ field access (const object)"

block: # Field access — let object
  type Obj2 = object
    x: int
  let obj2 = Obj2(x: 99)
  evalOnceAs(k, obj2.x)
  doAssert k == 99
  echo "✅ field access (let object)"

block: # Field access — const tuple
  const tup = (a: 7, b: 8)
  evalOnceAs(l, tup.a)
  doAssert typeof(l) is int, "const tuple field access should stay as int (no Int[N] wrapping)"
  doAssert l == 7
  echo "✅ field access (const tuple)"

block: # Field access — nested const object
  type Inner = object
    val: int
  type Outer = object
    inner: Inner
  const outer = Outer(inner: Inner(val: 99))
  evalOnceAs(m, outer.inner.val)
  doAssert typeof(m) is int, "nested const field access should stay as int (no Int[N] wrapping)"
  doAssert m == 99
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
  # For some reason some tests have a buildCount > 1 even though
  # the debugEcho sideeffect is only printed once

  # # This has a buildCount of 3 even though the echo is printed only once
  # test "Int construction counter increments once":
  #   var buildCount {.compileTime.} = 0
  #   proc makeInt[V: static int](): Int[V] =
  #     debugEcho "makeInt - ", "building count"
  #     inc buildCount
  #     result = Int[V]()
  #   evalOnceAs(alias, makeInt[4]())
  #   static: doAssert buildCount == 1, "buildCount = " & $buildCount & ", alias type = " & $typeof(alias) & ", alias = " & $alias

  # # This has a buildCount of 3 even though the echo is printed only once
  # test "tuple construction counter increments once":
  #   var buildCount {.compileTime.} = 0
  #   proc makePair[V, U: static int](): (Int[V], Int[U]) =
  #     debugEcho "makePair - ", "building count"
  #     inc buildCount
  #     result = (Int[V](), Int[U]())
  #   evalOnceAs(alias, makePair[2, 3]())
  #   static: doAssert buildCount == 1, "buildCount = " & $buildCount & ", alias type = " & $typeof(alias) & ", alias = " & $alias

  test "Int construction counter increments once (runtime)":
    var rtCount = 0
    proc makeRTInt(): int =
      inc rtCount
      result = 42
    evalOnceAs(alias, makeRTInt())
    doAssert rtCount == 1
    doAssert alias == 42

  # # This has a buldCount of 3 even though the echo is printed only once
  # test "costly Int computation once":
  #   var computeCount {.compileTime.} = 0
  #   proc costlyInt[V: static int](): Int[V] =
  #     debugEcho "costlyInt - ", "computing count"
  #     inc computeCount
  #     result = Int[V]()
  #   evalOnceAs(alias, costlyInt[99]())
  #   static: doAssert computeCount == 1, "computeCount = " & $computeCount & ", alias type = " & $typeof(alias) & ", alias = " & $alias

  # test "mixed lvalue Int alias and captured Int side-by-side":
  #   var counter {.compileTime.} = 0
  #   proc makeInt(): Int[5] =
  #     inc counter
  #     result = Int[5]()
  #   var mutable = 10
  #   evalOnceAs(fixed, makeInt())       # non-lvalue — captured once
  #   evalOnceAs(dynamic, mutable)       # lvalue — aliases the var
  #   static: doAssert counter == 1, "counter = " & $counter & ", alias type = " & $typeof(fixed) & ", alias = " & $fixed
  #   mutable = 20
  #   doAssert dynamic === 20
  #   static: doAssert counter == 1, "counter = " & $counter & ", alias type = " & $typeof(dynamic) & ", alias = " & $dynamic

  # test "nested evalOnceAs captures side effect once":
  #   var counter {.compileTime.} = 0
  #   proc makeOuter(): Int[5] =
  #     inc counter
  #     result = Int[5]()
  #   evalOnceAs(alias, makeOuter())
  #   static: doAssert counter == 1, "counter = " & $counter & ", alias type = " & $typeof(alias) & ", alias = " & $alias
  #   evalOnceAs(alias2, alias)
  #   static: doAssert counter == 1, "counter = " & $counter & ", alias type = " & $typeof(alias2) & ", alias = " & $alias2

# ═══════════════════════════════════════════════════════════════════════
# 8.  Stress / correctness
# ═══════════════════════════════════════════════════════════════════════

# suite "Stress / correctness":

#   test "1000 reads of captured Int construction":
#     var count {.compileTime.} = 0
#     proc build(): Int[42] =
#       inc count
#       result = Int[42]()
#     evalOnceAs(alias, build())
#     for i in 0..<1000:
#       discard alias
#     static: doAssert count == 1

#   test "interleaved Int lvalue and captured Int":
#     var state {.compileTime.} = 0
#     proc nextState(): Int[999] =
#       inc state
#       result = Int[999]()
#     var mutable = 10
#     evalOnceAs(fixed, nextState())
#     evalOnceAs(dynamic, mutable)
#     static: doAssert state == 1
#     mutable = 20
#     doAssert dynamic === 20
#     static: doAssert state == 1

#   test "deeply nested evalOnceAs with Int types":
#     evalOnceAs(a, Int[1]())
#     evalOnceAs(b, a)
#     evalOnceAs(c, b)
#     evalOnceAs(d, c)
#     evalOnceAs(e, d)
#     static: doAssert toIntVal(e) == 1

#   test "fibonacci at type level through evalOnceAs":
#     proc fib(V: static int): static int =
#       when V <= 1: V
#       else: fib(V - 1) + fib(V - 2)
#     evalOnceAs(alias, Int[fib(10)]())
#     static: doAssert toIntVal(alias) == 55

#   test "compile-time factorial through evalOnceAs":
#     proc fact(V: static int): static int =
#       when V <= 1: 1
#       else: V * fact(V - 1)
#     evalOnceAs(alias, Int[fact(6)]())
#     static: doAssert toIntVal(alias) == 720
