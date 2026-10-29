# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Integer-semimodule vocabulary.
##
## A stride element is any member of an integer-semimodule
## (associative `+`, scalar `·`, C26 Def 2.16):
## - plain `int` and `Int[N]` compile-time ints, the operator matrix
##   folds and decays them
## - coordinate-stride atoms, dispatched in `ism_coord_strides`
##
## A layout walk emits `+`/`*`/`div` over leaves and the overloads
## here dispatch the rest. NimNode emission leads the file,
## `getStaticInt` closes it.

import std/macros, std/typetraits
import workspace/math
import workspace/ceramic/src/macros/static_for

# ═══════════════════════════════════════════════════════════════════════
#  NimNode emission, natural arithmetic over AST nodes
# ═══════════════════════════════════════════════════════════════════════

func `*`*(a, b: NimNode): NimNode {.compileTime.} =
  nnkInfix.newTree(ident"*", a, b)

func `div`*(a, b: NimNode): NimNode {.compileTime.} =
  nnkInfix.newTree(ident"div", a, b)

func `+`*(a, b: NimNode): NimNode {.compileTime.} =
  nnkInfix.newTree(ident"+", a, b)

func `-`*(a, b: NimNode): NimNode {.compileTime.} =
  nnkInfix.newTree(ident"-", a, b)

func `mod`*(a, b: NimNode): NimNode {.compileTime.} =
  nnkInfix.newTree(ident"mod", a, b)

func abs*(a: NimNode): NimNode {.compileTime.} =
  bindSym"abs".newCall(a)

func min*(a, b: NimNode): NimNode {.compileTime.} =
  bindSym"min".newCall(a, b)

func max*(a, b: NimNode): NimNode {.compileTime.} =
  bindSym"max".newCall(a, b)

func ceil_div*(a, b: NimNode): NimNode {.compileTime.} =
  bindSym"ceil_div".newCall(a, b)

func round_multiple_up*(a, b: NimNode): NimNode {.compileTime.} =
  bindSym"round_multiple_up".newCall(a, b)

func sign*(a: NimNode): NimNode {.compileTime.} =
  bindSym"sign".newCall(a)


const DynamicSentinel* = low(int)
  ## Sentinel value used throughout the library to mark "unknown at compile time" (dynamic/runtime int).
  ## 0-stride instead can be confused with a broadcasted dimension.

# ═══════════════════════════════════════════════════════════════════════
#  Int[N]
# ═══════════════════════════════════════════════════════════════════════
#
# An Int phantom type propagates values across Ceramic without losing const-ness at function boundaries.
# Low-level primitives and overloads are templates: direct substitution, no generated-code indirection.
# A template input must appear once in the body or the input inlines twice, side-effects included.
# An input of 'echo "launch missiles"' prints per inlining.

type
  Int*[V: static int] = object
    ## Compile-time integer literal
    ## This allows constant-folding as plain 'int' lose constness across function boundaries

  IntOrIntTuple* = int | Int | tuple

template `$`*[V: static int](x: Int[V]): string = "_" & $V

func `==`*[V: static int](a: Int[V], b: int): bool {.error: "`==` is not defined for Int. If this comparison is intentional, please use `===`".}
func `==`*[V: static int](a: int, b: Int[V]): bool {.error: "`==` is not defined for Int. If this comparison is intentional, please use `===`".}
func `==`*[V, U: static int](a: Int[V], b: Int[U]): bool {.error: "`==` is not defined for Int. If this comparison is intentional, please use `===`".}

template rank*(t: typedesc[IntOrIntTuple]): static int =
  when t is (int or Int):
    1
  else:
    tupleLen(t)

template rank*(t: IntOrIntTuple): static int =
  when t is (int or Int):
    1
  else:
    tupleLen(typeof(t))

# ═══════════════════════════════════════════════════════════════════════
#  Int[N] ordering, global overloads for tuple comparison
# ═══════════════════════════════════════════════════════════════════════

template genCmpOp(op: untyped): untyped =
  template op*[V, U: static int](a: Int[V], b: Int[U]): static bool = op(V, U)
  template op*[V: static int](a: Int[V], b: static int): static bool = op(V, b)
  template op*[V: static int](a: static int, b: Int[V]): static bool = op(a, V)
  template op*[V: static int](a: Int[V], b: int): bool = op(V, b)
  template op*[V: static int](a: int, b: Int[V]): bool = op(a, V)

genCmpOp(`<`)
genCmpOp(`<=`)
genCmpOp(`>`)
genCmpOp(`>=`)

# ═══════════════════════════════════════════════════════════════════════
#  `===`, deep element-wise comparison across Int[N] and int
# ═══════════════════════════════════════════════════════════════════════

template `===`*(a, b: int): bool = a == b
template `===`*(a, b: static int): bool = a == b
template `===`*[V, U: static int](a: Int[V], b: Int[U]): bool = V == U

template `===`*[V: static int](a: Int[V], b: int): bool = V == b
template `===`*[V: static int](a: int, b: Int[V]): bool = a == V
template `===`*[V: static int](a: Int[V], b: static int): bool = V == b
template `===`*[V: static int](a: static int, b: Int[V]): bool = a == V

func `===`*[T: tuple, U: tuple](a: T, b: U): bool {.inline.} =
  ## Deep element-wise tuple comparison.
  when tupleLen(T) != tupleLen(U):
    false
  else:
    staticFor i, 0, tupleLen(T):
      if not (a[i] === b[i]):
        return false
    true

template `===`*[T: tuple](a: T, b: int): bool =
  ## Compare a tuple against an int, valid for 1-element tuples only.
  when tupleLen(T) == 1:
    a[0] === b
  else:
    false

template `===`*[V: static int](a: Int[V], b: tuple): bool =
  ## An Int equals a 1-element tuple holding an equal element.
  when tupleLen(b) == 1:
    a === b[0]
  else:
    false

template `===`*[T: tuple, V: static int](a: T, b: Int[V]): bool =
  ## A tuple equals an Int only as a 1-element tuple.
  when tupleLen(T) == 1:
    a[0] === b
  else:
    false

template `===`*[U: tuple](a: int, b: U): bool =
  ## Compare an int against a tuple, valid for 1-element tuples only.
  when tupleLen(U) == 1:
    a === b[0]
  else:
    false

# ═══════════════════════════════════════════════════════════════════════
#  `!==`, negation of the deep element-wise comparison
# ═══════════════════════════════════════════════════════════════════════

template `!==`*(a, b: auto): bool = not (a === b)

# ═══════════════════════════════════════════════════════════════════════
#  Int[N] arithmetic
# ═══════════════════════════════════════════════════════════════════════

template sign*[V: static int](x: Int[V]): auto = Int[sign(V)]()

template abs*[V: static int](x: Int[V]): auto = Int[abs(V)]()

template genBinOp(op: untyped): untyped =
  template op*[V, U: static int](a: Int[V], b: Int[U]): auto = Int[op(V, U)]()
  template op*[V: static int](a: Int[V], b: static int): auto = Int[op(V, b)]()
  template op*[V: static int](a: static int, b: Int[V]): auto = Int[op(a, V)]()
  template op*[V: static int](a: Int[V], b: int): int = op(V, b)
  template op*[V: static int](a: int, b: Int[V]): int = op(a, V)

genBinOp(`+`)
genBinOp(`-`)
genBinOp(`*`)
genBinOp(`div`)
genBinOp(`mod`)

genBinOp(`max`)
genBinOp(`min`)
genBinOp(`ceil_div`)
genBinOp(`round_multiple_up`)
genBinOp(`round_multiple_down`)
genBinOp(`gcd`)

template `+=`*[V: static int](a: var int, b: Int[V]) = a += V

# Specialized overloads
template `*`*(a: Int[0], b: int): Int[0] = Int[0]()
template `*`*(a: int, b: Int[0]): Int[0] = Int[0]()

# ═══════════════════════════════════════════════════════════════════════
#  iteration bounds
# ═══════════════════════════════════════════════════════════════════════

template `..<`*[V: static int](start: int, bound: Int[V]): Slice[int] =
  Slice[int](a: start, b: pred(V))
template `..<`*[V: static int](start: Int[V], bound: int): Slice[int] =
  Slice[int](a: V, b: pred(bound))
template `..<`*[V, U: static int](start: Int[V], bound: Int[U]): Slice[int] =
  Slice[int](a: V, b: pred(U))

# ═══════════════════════════════════════════════════════════════════════
#  countup over mixed bounds
# ═══════════════════════════════════════════════════════════════════════

template countup*[V: static int](start: int, stopIncl: Int[V], step: int = 1): untyped =
  countup(start, V, step)

template countup*[V: static int](start: Int[V], stopIncl: int, step: int = 1): untyped =
  countup(V, stopIncl, step)

template countup*[V, U: static int](start: Int[V], stopIncl: Int[U], step: int = 1): untyped =
  countup(V, U, step)

# ═══════════════════════════════════════════════════════════════════════
#  Int[N] compile-time helpers for macros
# ═══════════════════════════════════════════════════════════════════════

func IntCT*(val: int): NimNode {.compileTime.} =
  ## Shorthand: Int[val]() AST node.
  newNimNode(nnkObjConstr).add(newNimNode(nnkBracketExpr).add(ident"Int", newLit(val)))

func intTy*(val: int): NimNode {.compileTime.} =
  ## Int[val] bracket for the compile-time value `val`
  IntCT(val)[0]

func divmod*(a, b: int): (int, int) {.compileTime.} =
  ## Quotient and remainder of `a` divided by `b`.
  (a div b, a mod b)

func isStaticInt*(t: NimNode): bool {.compileTime.} =
  (t.kind == nnkBracketExpr and $t[0] == "Int") or t.kind == nnkIntLit

func isStaticOne*(t: NimNode): bool {.compileTime.} =
  (t.kind == nnkBracketExpr and $t[0] == "Int" and t[1].intVal == 1) or
  (t.kind == nnkIntLit and t.intVal == 1)

func isIntTyHead*(n: NimNode): bool {.compileTime.} =
  n.kind == nnkSym and (n.sameType(bindSym"Int") or $n == "Int")

func isScalarTy*(t: NimNode): bool {.compileTime.} =
  t.kind == nnkIntLit or
    (t.kind == nnkBracketExpr and t[0].isIntTyHead()) or
    (t.kind == nnkCall and t[0].kind == nnkBracketExpr and t[0][0].isIntTyHead()) or
    (t.kind == nnkSym and t.sameType(bindSym"int"))

func getStaticInt*(t: NimNode): int {.compileTime.} =
  ## Static Int value of a node.
  ## - the result is DynamicSentinel when the node lacks a static Int:
  ##   not a literal, not an `Int[V]` type or construction
  ##
  ## TODO: duplicating Nim constant-folding via Int overload is unwanted.
  case t.kind
  of nnkIntLit, nnkUIntLit:
    int(t.intVal)
  of nnkStmtListExpr:
    # evalOnceAs caching, `const q = X; X`, the trailing value is static
    t[^1].getStaticInt()
  of nnkCall, nnkBracketExpr:
    # typeDesc[Int[V]] wrappers: the typed-delegate path produces these
    if t[0].eqIdent("typeDesc"):
      t[1].getStaticInt()
    elif t[0].eqIdent("Int"):
      t[1].getStaticInt()
    elif t[0].eqIdent("ceil_div") and t.len == 3:
      # A `ceil_div` call over two static operands folds to its value.
      # Nim applies the same folding to static ints during type resolution.
      let a = t[1].getStaticInt()
      let b = t[2].getStaticInt()
      if a == DynamicSentinel or b == DynamicSentinel or b == 0:
        DynamicSentinel
      else:
        math.ceil_div(a, b)
    elif t[0].eqIdent("round_multiple_up") and t.len == 3 and
        t[1].kind in {nnkIntLit, nnkUIntLit} and t[2].kind in {nnkIntLit, nnkUIntLit}:
      let a = t[1].intVal.int
      let b = t[2].intVal.int
      if b == 0:
        DynamicSentinel
      else:
        math.round_multiple_up(a, b)
    elif t[0].eqIdent("round_multiple_down") and t.len == 3 and
        t[1].kind in {nnkIntLit, nnkUIntLit} and t[2].kind in {nnkIntLit, nnkUIntLit}:
      let a = t[1].intVal.int
      let b = t[2].intVal.int
      if b == 0:
        DynamicSentinel
      else:
        math.round_multiple_down(a, b)
    else:
      DynamicSentinel
  of nnkInfix:
    # Algebra emission produces natural arithmetic such as `Int[a] div Int[b]`,
    # folded here one operator per step, before Nim semchecks the expression.
    let a = t[1].getStaticInt()
    let b = t[2].getStaticInt()
    if a == DynamicSentinel or b == DynamicSentinel:
      return DynamicSentinel
    case $t[0]
    of "+":
      a + b
    of "-":
      a - b
    of "*":
      a * b
    of "div":
      if b != 0:
        a div b
      else:
        DynamicSentinel
    of "mod":
      if b != 0:
        a mod b
      else:
        DynamicSentinel
    else:
      DynamicSentinel
  of nnkObjConstr:
    if t[0].kind == nnkBracketExpr and t[0][0].eqIdent("Int"):
      t[0].getStaticInt()
    else:
      DynamicSentinel
  of nnkSym:
    let ti = t.getTypeInst()
    if ti.kind != nnkSym:
      ti.getStaticInt()
    else:
      DynamicSentinel
  else:
    DynamicSentinel
