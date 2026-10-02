# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.


import std/macros, std/typetraits, std/math
import workspace/ceramic/src/macros/static_for

# ═══════════════════════════════════════════════════════════════
#  Int[N]
# ═══════════════════════════════════════════════════════════════
#
# We provide an Int phantom type that will be used to propagate values across Ceramic
# without them losing their const-ness at function boundaries.
#
# Furthermore low-level primitives and overload are provided as templates for most.
#
# Templates are substituted directly without generated code, significantly improving readability
# of generated code as there is no function indirection.
#
# A limitation is that inputs MUST be used only once in the template body or inputs will be inlined twice, including their side-effects.
# Concretely if an input that does 'echo "launch missiles"' is passed to a template, "launch missiles" will be printed as many times as the input is called within the template.

type
  Int*[V: static int] = object
    ## Compile-time integer literal
    ## This allows constant-folding as plain 'int' lose constness across function boundaries

  IntOrIntTuple* = int | Int | tuple

template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

template `$`*[V: static int](x: Int[V]): string = "Int[" & $V & "]"

func `==`*[V: static int](a: Int[V]; b: int): bool {.error: "`==` is not defined for Int. If this comparison is intentional, please use `===`".}
func `==`*[V: static int](a: int; b: Int[V]): bool {.error: "`==` is not defined for Int. If this comparison is intentional, please use `===`".}
func `==`*[V, U: static int](a: Int[V]; b: Int[U]): bool {.error: "`==` is not defined for Int. If this comparison is intentional, please use `===`".}

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

# ═══════════════════════════════════════════════════════════════
#  Int[N] == int, global overloads for tuple comparison
# ═══════════════════════════════════════════════════════════════

template `<=`*[V: static int](a: Int[V]; b: int): bool = V <= b
template `<=`*[V: static int](a: int; b: Int[V]): bool = a <= V
template `>=`*[V: static int](a: Int[V]; b: int): bool = V >= b
template `>=`*[V: static int](a: int; b: Int[V]): bool = a >= V
template `<=`*[V, U: static int](a: Int[V]; b: Int[U]): static bool = V <= U
template `>=`*[V, U: static int](a: Int[V]; b: Int[U]): static bool = V >= U

# ═══════════════════════════════════════════════════════════════
#  `===`, deep element-wise comparison across Int[N] and int
# ═══════════════════════════════════════════════════════════════

template `===`*(a, b: int): bool = a == b
template `===`*(a, b: static int): bool = a == b
template `===`*[V, U: static int](a: Int[V]; b: Int[U]): bool = V == U

template `===`*[V: static int](a: Int[V]; b: int): bool = V == b
template `===`*[V: static int](a: int; b: Int[V]): bool = a == V
template `===`*[V: static int](a: Int[V]; b: static int): bool = V == b
template `===`*[V: static int](a: static int; b: Int[V]): bool = a == V

func `===`*[T: tuple, U: tuple](a: T; b: U): bool {.inline.} =
  ## Deep element-wise tuple comparison.
  when tupleLen(T) != tupleLen(U):
    false
  else:
    staticFor i, 0, tupleLen(T):
      if not (a[i] === b[i]):
        return false
    true

template `===`*[T: tuple](a: T; b: int): bool =
  ## Compare a tuple against an int — only valid for 1-element tuples.
  when tupleLen(T) == 1:
    a[0] === b
  else:
    false

template `===`*[U: tuple](a: int; b: U): bool =
  ## Compare an int against a tuple — only valid for 1-element tuples.
  when tupleLen(U) == 1:
    a === b[0]
  else:
    false

# ═══════════════════════════════════════════════════════════════
#  `!==`, negation of the deep element-wise comparison
# ═══════════════════════════════════════════════════════════════

template `!==`*(a, b: auto): bool = not (a === b)

# ═══════════════════════════════════════════════════════════════
#  Int[N] arithmetic
# ═══════════════════════════════════════════════════════════════

func ceil_div*(a, b: int): int {.inline.} =
  (a + b - 1) div b

func ceil_div*(a, b: static int): static int {.inline.} =
  ## Static overload, both arguments fold at compile time.
  (a + b - 1) div b

func sign*(x: int): int {.inline.} =
  if x > 0: 1 elif x < 0: -1 else: 0

func sign*(x: static int): static int {.inline.} =
  ## Static overload, folds at compile time.
  if x > 0: 1 elif x < 0: -1 else: 0

template sign*[V: static int](x: Int[V]): auto =
  const S =
    if V > 0: 1
    elif V < 0: -1
    else: 0
  Int[S]()

template abs*[V: static int](x: Int[V]): auto = Int[abs(V)]()

template genBinOp(op: untyped): untyped =
  template op*[V, U: static int](a: Int[V]; b: Int[U]): auto = Int[op(V, U)]()
  template op*[V: static int](a: Int[V]; b: static int): auto = Int[op(V, b)]()
  template op*[V: static int](a: static int; b: Int[V]): auto = Int[op(a, V)]()
  template op*[V: static int](a: Int[V]; b: int): int = op(V, b)
  template op*[V: static int](a: int; b: Int[V]): int = op(a, V)

genBinOp(`+`)
genBinOp(`-`)
genBinOp(`*`)
genBinOp(`div`)
genBinOp(`mod`)

genBinOp(`max`)
genBinOp(`min`)
genBinOp(`ceil_div`)
genBinOp(`gcd`)

template `+=`*[V: static int](a: var int; b: Int[V]) = a += V

# ═══════════════════════════════════════════════════════════════
#  iteration bounds
# ═══════════════════════════════════════════════════════════════

template `..<`*[V: static int](start: int; bound: Int[V]): Slice[int] =
  Slice[int](a: start, b: pred(V))
template `..<`*[V: static int](start: Int[V]; bound: int): Slice[int] =
  Slice[int](a: V, b: pred(bound))
template `..<`*[V, U: static int](start: Int[V]; bound: Int[U]): Slice[int] =
  Slice[int](a: V, b: pred(U))
