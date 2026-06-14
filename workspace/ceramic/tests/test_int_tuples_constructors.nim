# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/ceramic/src/int_tuples {.all.}
import std/unittest

# ═══════════════════════════════════════════════════════════════
#  makeIntTuple constructor tests
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  Helper procs used in tests below
# ═══════════════════════════════════════════════════════════════

proc identity[T](x: T): T = x
template first(t: untyped): untyped = t[0]
template second(t: untyped): untyped = t[1]
template tplIdentity(x: untyped): untyped = x
template tplWrap(x: untyped): untyped = (x,)

proc runTests() =

  suite "makeIntTuple — convert int leaves to Int[N]":

    test "tuple of ints":
      let t = makeIntTuple((3, 4))
      check t === (Int[3](), Int[4]())
      doAssert t is (Int[3], Int[4])

    test "1-element tuple":
      let t = makeIntTuple((5,))
      check t === (Int[5](),)
      doAssert t is (Int[5],)

    test "nested tuple":
      let t = makeIntTuple(((1, 2), (3,)))
      check t === ((Int[1](), Int[2]()), (Int[3](),))
      doAssert t is ((Int[1], Int[2]), (Int[3],))

    test "trivially nested — 3-element":
      let t = makeIntTuple((10, 20, 30))
      check t === (Int[10](), Int[20](), Int[30]())
      doAssert t is (Int[10], Int[20], Int[30])

    test "mixed int and Int[N] leaves":
      let t = makeIntTuple((1, Int[5](), 3))
      check t === (Int[1](), Int[5](), Int[3]())
      doAssert t is (Int[1], Int[5], Int[3])

    test "deeply nested":
      let t = makeIntTuple(((1, (2, 3)), 4))
      check t === ((Int[1](), (Int[2](), Int[3]())), Int[4]())
      doAssert t is ((Int[1], (Int[2], Int[3])), Int[4])

    test "empty tuple":
      discard makeIntTuple(())

  suite "let / const assignment indirection":

    test "let indirection — (3, 4) keeps int runtime":
      # `let` is runtime — static int info is lost.
      let src = (3, 4)
      let t = makeIntTuple(src)
      check t[0] == 3 and t[1] == 4
      doAssert t is (int, int)

    test "let indirection — nested ((1, 2), (3,)) keeps int":
      let src = ((1, 2), (3,))
      let t = makeIntTuple(src)
      check t[0][0] == 1 and t[0][1] == 2 and t[1][0] == 3
      doAssert t is ((int, int), (int,))

    test "const indirection — (3, 4)":
      const src = (3, 4)
      let t = makeIntTuple(src)
      check t === (Int[3](), Int[4]())
      doAssert t is (Int[3], Int[4])

    test "const indirection — nested ((1, 2), (3,))":
      const src = ((1, 2), (3,))
      let t = makeIntTuple(src)
      check t === ((Int[1](), Int[2]()), (Int[3](),))
      doAssert t is ((Int[1], Int[2]), (Int[3],))

    test "const indirection — 3-element":
      const src = (10, 20, 30)
      let t = makeIntTuple(src)
      check t === (Int[10](), Int[20](), Int[30]())
      doAssert t is (Int[10], Int[20], Int[30])

  var runtimeCounter = 0
  proc runtimeInt(): int =
    inc runtimeCounter
    result = runtimeCounter

  suite "Proc calls on let / const variables":

    test "let tuple through identity proc — stays int":
      # `let` is runtime — no conversion.
      let src = (3, 4)
      let t = makeIntTuple(identity(src))
      check t[0] == 3 and t[1] == 4
      doAssert t is (int, int)

    test "const tuple through identity proc":
      const src = (3, 4)
      let t = makeIntTuple(identity(src))
      check t === (Int[3](), Int[4]())
      doAssert t is (Int[3], Int[4])

    test "let -> proc -> first element — stays int":
      let src = (3, 4)
      let t = makeIntTuple(first(src))
      check t == 3
      doAssert t is int

    test "const -> proc -> second element":
      const src = (10, 20)
      let t = makeIntTuple(second(src))
      check t === Int[20]()
      doAssert t is Int[20]

  suite "Constant folding — chained proc calls":

    proc double[V: static int](x: Int[V]): Int[V * 2] = Int[V * 2]()
    proc add[V, U: static int](a: Int[V]; b: Int[U]): Int[V + U] = Int[V + U]()

    test "chained procs: add(double(Int[3]), double(Int[4]))":
      let t = makeIntTuple((add(double(Int[3]()), double(Int[4]())),))
      check t === (Int[14](),)  # (3*2)+(4*2) = 14
      doAssert t is (Int[14],)

    test "identity through procs on const tuple":
      const src = (5, 6, 7)
      let t = makeIntTuple(identity(src))
      check t === (Int[5](), Int[6](), Int[7]())
      doAssert t is (Int[5], Int[6], Int[7])

    test "proc chain on let tuple with Int[N] leaf":
      # Second element is truly runtime int, must stay int.
      let src = (Int[3](), runtimeInt())
      let t = makeIntTuple(src)
      check t[0] === Int[3]()
      check t[1] == 1
      doAssert t is (Int[3], int)

  suite "Template calls on const variables":

    test "const tuple through template":
      const src = (7, 8)
      let t = makeIntTuple(tplIdentity(src))
      check t === (Int[7](), Int[8]())
      doAssert t is (Int[7], Int[8])

    test "const tuple wrapped in 1-tuple via template":
      const src = (3, 4)
      let t = makeIntTuple(tplWrap(src))
      check t === ((Int[3](), Int[4]()),)
      doAssert t is ((Int[3], Int[4]),)

    test "let tuple through template chain with runtime int":
      # Both elements are runtime ints — no conversion.
      var x = 10
      let src = ((9, x),)
      let t = makeIntTuple(tplIdentity(tplIdentity(src)))
      check t[0][0] == 9
      check t[0][1] == 10
      doAssert t is ((int, int),)

  suite "Proc on literal — const folding across proc boundaries":

    proc identityInt(x: int): int = x
    proc identityGeneric[T](x: T): T = x

    test "non-generic int proc with literal — identityInt(42)":
      let t = makeIntTuple(identityInt(42))
      check t === Int[42]()
      doAssert t is Int[42]

    test "generic proc with literal — identityGeneric(42)":
      let t = makeIntTuple(identityGeneric(42))
      check t === Int[42]()
      doAssert t is Int[42]

    test "proc result captured in let — runtime, no conversion":
      let x = identityInt(42)
      let t = makeIntTuple(x)
      check t == 42
      doAssert t is int

    test "chained non-generic procs with literal":
      let t = makeIntTuple(identityInt(identityInt(42)))
      check t === Int[42]()
      doAssert t is Int[42]

    test "identityGeneric through identityInt — still const-folded":
      let t = makeIntTuple(identityInt(identityGeneric(42)))
      check t === Int[42]()
      doAssert t is Int[42]
  suite "Const objects and int literals":

    type
      Pair = object
        a: int
        b: int

    proc toTuple(p: Pair): (int, int) = (p.a, p.b)

    test "const Pair object":
      const p = Pair(a: 3, b: 4)
      let t = makeIntTuple(toTuple(p))
      check t === (Int[3](), Int[4]())
      doAssert t is (Int[3], Int[4])

    test "let Pair object — stays int":
      # `let` is runtime — no conversion.
      let p = Pair(a: 10, b: 20)
      let t = makeIntTuple(toTuple(p))
      check t[0] == 10 and t[1] == 20
      doAssert t is (int, int)

    test "scalar int literal":
      let t = makeIntTuple(42)
      check t === Int[42]()
      doAssert t is Int[42]

    test "const int":
      const c = 99
      let t = makeIntTuple(c)
      check t === Int[99]()
      doAssert t is Int[99]

    test "negative int literal":
      let t = makeIntTuple(-5)
      check t === Int[-5]()
      doAssert t is Int[-5]

    test "const scalar through identity":
      const c = 42
      let t = makeIntTuple(identity(c))
      check t === Int[42]()
      doAssert t is Int[42]

    test "Int[N] passthrough":
      let t = makeIntTuple(Int[42]())
      check t === Int[42]()
      doAssert t is Int[42]

when isMainModule:
  runTests()