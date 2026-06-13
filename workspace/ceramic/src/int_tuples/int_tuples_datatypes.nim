# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.


import std/macros, std/typetraits
import ./macros/static_for

# ═══════════════════════════════════════════════════════════════
#  Int[N] — compile-time integer type
# ═══════════════════════════════════════════════════════════════

type Int*[V: static int] = object
  ## Compile-time integer literal, analogous to CuTe's `Int<N>`.
  ##   Int<4>  ⇔  Int[4]

type IntOrIntTuple* = int | Int | tuple
  ## Shape/stride element type alias for convenience.

func toIntVal*(x: int): int = x
func toIntVal*[V: static int](x: Int[V]): int = V

func `$`*[V: static int](x: Int[V]): static string = "Int[" & $V & "]"

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
  func op*[V, U: static int](a: Int[V]; b: Int[U]): Int[op(V, U)] = Int[op(V, U)]()
  func op*[V: static int](a: Int[V]; b: static int): Int[op(V, b)] = Int[op(V, b)]()
  func op*[V: static int](a: static int; b: Int[V]): Int[op(a, V)] = Int[op(a, V)]()
  func op*[V: static int](a: Int[V]; b: int): int = op(V, b)
  func op*[V: static int](a: int; b: Int[V]): int = op(a, V)

genBinOp(`+`)
genBinOp(`-`)
genBinOp(`*`)
genBinOp(`div`)
genBinOp(`mod`)

genBinOp(`max`)
genBinOp(`min`)
genBinOp(`ceil_div`)

func `+=`*[V: static int](a: var int; b: Int[V]) = a += V
