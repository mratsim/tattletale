# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.


import std/macros, std/typetraits
import ../macros/static_for

# ═══════════════════════════════════════════════════════════════
#  Int[N] — compile-time integer type
# ═══════════════════════════════════════════════════════════════

type Int*[V: static int] = object
  ## Compile-time integer literal, analogous to CuTe's `Int<N>`.
  ##   Int<4>  ⇔  Int[4]

type IntOrIntTuple* = int | Int | tuple
  ## Shape/stride element type alias for convenience.

## ── PERF CRITICAL: templates, not funcs ──
## toIntVal(int) is called on EVERY element access (via []/() operators).
## As a non-`{.inline.}` `func`, it lands in its own C++ compilation unit,
## blocking cross-module inlining. Even `{.inline.}` generates a C++ function
## definition the inliner must process; `template` produces zero C++ definitions.
## When extracting Int[V] → int, a template extracts V directly in the generated
## C++ expression; a func wraps it in a C struct parameter that is noisy for no gain.
## History: commit b93bb95 changed func→template, recovering ~12× on flat-index copies.

template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V


func `$`*[V: static int](x: Int[V]): static string = "Int[" & $V & "]"

func `==`*[V: static int](a: Int[V]; b: int): bool {.error: "`==` is not defined for Int. If this comparison is intentional, please use `===`".}
func `==`*[V: static int](a: int; b: Int[V]): bool {.error: "`==` is not defined for Int. If this comparison is intentional, please use `===`".}
func `==`*[V, U: static int](a: Int[V]; b: Int[U]): bool {.error: "`==` is not defined for Int. If this comparison is intentional, please use `===`".}

func rank*(t: typedesc[IntOrIntTuple]): static int =
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
#  Int[N] == int — global overloads for tuple comparison
# ═══════════════════════════════════════════════════════════════

func `<=`*[V: static int](a: Int[V]; b: int): bool {.inline.} = V <= b
func `<=`*[V: static int](a: int; b: Int[V]): bool {.inline.} = a <= V
func `>=`*[V: static int](a: Int[V]; b: int): bool {.inline.} = V >= b
func `>=`*[V: static int](a: int; b: Int[V]): bool {.inline.} = a >= V
func `<=`*[V, U: static int](a: Int[V]; b: Int[U]): static bool = V <= U
func `>=`*[V, U: static int](a: Int[V]; b: Int[U]): static bool = V >= U

# ═══════════════════════════════════════════════════════════════
#  `===` — deep element-wise tuple comparison (handles Int[N] vs int)
# ═══════════════════════════════════════════════════════════════

func `===`*(a, b: int): bool {.inline.} = a == b
func `===`*(a, b: static int): static bool = a == b
func `===`*[V, U: static int](a: Int[V]; b: Int[U]): static bool = V == U

func `===`*[V: static int](a: Int[V]; b: int): bool {.inline.} = V == b
func `===`*[V: static int](a: int; b: Int[V]): bool {.inline.} = a == V
func `===`*[V: static int](a: Int[V]; b: static int): static bool = V == b
func `===`*[V: static int](a: static int; b: Int[V]): static bool = a == V

func `===`*[T: tuple, U: tuple](a: T; b: U): bool {.inline.} =
  ## Deep element-wise tuple comparison.
  ## Handles Int[N] vs int mismatches via per-element === overloads.
  when tupleLen(T) != tupleLen(U):
    false
  else:
    staticFor i, 0, tupleLen(T):
      if not (a[i] === b[i]):
        return false
    true

func `===`*[T: tuple](a: T; b: int): bool {.inline.} =
  ## Compare a tuple against an int — only valid for 1-element tuples.
  when tupleLen(T) == 1:
    a[0] === b
  else:
    false

func `===`*[U: tuple](a: int; b: U): bool {.inline.} =
  ## Compare an int against a tuple — only valid for 1-element tuples.
  when tupleLen(U) == 1:
    a === b[0]
  else:
    false

# ═══════════════════════════════════════════════════════════════
#  `!==` — negation of deep element-wise tuple comparison
# ═══════════════════════════════════════════════════════════════

func `!==`*(a, b: auto): bool {.inline.} = not (a === b)

# ═══════════════════════════════════════════════════════════════
#  Int[N] arithmetic
# ═══════════════════════════════════════════════════════════════

func ceil_div*(a, b: int): int =
  (a + b - 1) div b

func abs*[V: static int](x: Int[V]): Int[abs(V)] = Int[abs(V)]()

func sign*[V: static int](x: Int[V]): Int[if V > 0: 1 elif V < 0: -1 else: 0] = discard

template genBinOp(op: untyped): untyped =
  ## Generate arithmetic operators for Int[V] vs int/Int.
  ##
  ## PERF CRITICAL: `Int[V] * int` and `int * Int[V]` MUST be `template`, not `func`.
  ## These are called inside crd2idx's foldZipWith inner product loop.
  ## A `func` with `Int[V]` arguments generates C struct-object parameters
  ## that the C++ inliner must unravel. A `template` collapses
  ## `Int[16]() * i` to the bare constant `16 * i` at the Nim codegen level.

  template op*[V, U: static int](a: Int[V]; b: Int[U]): auto = Int[op(V, U)]()
  func op*[V: static int](a: Int[V]; b: static int): auto {.inline.} = Int[op(V, b)]()
  func op*[V: static int](a: static int; b: Int[V]): auto {.inline.} = Int[op(a, V)]()
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

func `+=`*[V: static int](a: var int; b: Int[V]) = a += V

# ═══════════════════════════════════════════════════════════════
#  Iteration bounds
# ═══════════════════════════════════════════════════════════════

template `..<`*[V: static int](start: int; bound: Int[V]): Slice[int] =
  Slice[int](a: start, b: pred(V))
template `..<`*[V: static int](start: Int[V]; bound: int): Slice[int] =
  Slice[int](a: V, b: pred(bound))
template `..<`*[V, U: static int](start: Int[V]; bound: Int[U]): Slice[int] =
  Slice[int](a: V, b: pred(U))
