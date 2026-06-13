# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/typetraits
import ./int_tuples_datatypes
import ./int_tuples_transforms

# ═══════════════════════════════════════════════════════════════
#  fold — left-fold reduction with Int[N] support
# ═══════════════════════════════════════════════════════════════
#
#  Adapted from the pattern in int_tuples.nim.bak:
#    - Scalar `int` → inject `acc`, `it`, evaluate body
#    - Scalar `Int[N]` → inject `acc`, `it` (Int[V] → int via * overloads)
#    - Tuple → recurse over fields via `for f in fields(t)`
#
#  Injects `acc` (accumulator, type int) and `it` (current element).
#  `body` returns new accumulator.  Always returns `int`.
#
#  Examples:
#    fold(5, 1, acc * it)            → 5
#    fold(Int[5](), 1, acc * it)     → 5  (Int[V] extracted via * overload)
#    fold((2,3,4), 1, acc * it)      → 24
#    fold((2,(3,4)), 1, acc * it)    → 24
# ═══════════════════════════════════════════════════════════════

template fold_recurse*(idx: static int; t: tuple; state: typed; body: untyped): auto =
  let field = fold(t[idx], state, body)
  when idx == tupleLen(t) - 1:
    field
  else:
    fold_recurse(idx + 1, t, field, body)

template fold*(t: IntOrIntTuple; startingAcc: typed; body: untyped): auto =
  ## Fold over all leaves of t with an accumulator.
  ## Sub-tuples are handled recursively via fold_recurse.
  ## Returns Int[N] for all-Int[N] leaf paths, int otherwise.
  when t is int or t is Int:
    block:
      let acc {.inject.} = startingAcc
      let it {.inject.} = t
      body
  else:  # tuple
    when tupleLen(t) == 0:
      startingAcc
    else:
      fold_recurse(0, t, startingAcc, body)

# ═══════════════════════════════════════════════════════════════
#  prefix_scanIt / suffix_scanIt - scans while preserving constness
# ═══════════════════════════════════════════════════════════════
#
#  Recursive template block + concat for type-correct tuple building.
#  Injects `acc` (accumulator before element) and `it` (element).
#  Returns tuple where each element = `acc` BEFORE that element.
# ═══════════════════════════════════════════════════════════════

template tail_accumulator(strides, shape: IntOrIntTuple): auto =
  ## Final accumulator after prefix_scan: walks last-elem chain to the leaf.
  const L = tupleLen(typeof(shape)) - 1
  when shape[L] is int or shape[L] is Int:
    strides[L] * shape[L]
  else:
    tail_accumulator(strides[L], shape[L])

template head_accumulator(strides, shape: IntOrIntTuple): auto =
  ## Final accumulator after suffix_scan: walks first-elem chain to the leaf.
  when shape[0] is int or shape[0] is Int:
    strides[0] * shape[0]
  else:
    head_accumulator(strides[0], shape[0])

template prefix_scanIt_recurse*(idx: static int; t: tuple; state: typed; body: untyped): untyped =
  ## Recursive prefix scan. Each level injects acc/it into a block scope.
  ##
  ## Due to generic sandwich / template symbol resolution issues
  ## this is exported but it really is an internal module

  when t[idx] is tuple:
    let it = t[idx]
    let acc = state
    let subStrides = prefix_scanIt(it, acc, body)
    const L = tupleLen(typeof(it)) - 1
    let newState =
      when it[L] is int or it[L] is Int:
        subStrides[L] * it[L]
      else:
        tail_accumulator(subStrides[L], it[L], body)
    when idx == tupleLen(t) - 1:
      (subStrides,)
    else:
      concat((subStrides,), prefix_scanIt_recurse(idx + 1, t, newState, body))
  else:
    block:
      let it {.inject.} = t[idx]
      let acc {.inject.} = state
      let newState = body
      when idx == tupleLen(t) - 1:
        (acc,)
      else:
        concat((acc,), prefix_scanIt_recurse(idx + 1, t, newState, body))

template suffix_scanIt_recurse*(idx: static int; t: tuple; state: typed; body: untyped): untyped =
  ## Recursive suffix scan. Each level injects acc/it into a block scope.
  ##
  ## Due to generic sandwich / template symbol resolution issues
  ## this is exported but it really is an internal module

  when t[idx] is tuple:
    let it = t[idx]
    let acc = state
    let subStrides = suffix_scanIt(it, acc, body)
    let newState =
      when it[0] is int or it[0] is Int:
        subStrides[0] * it[0]
      else:
        head_accumulator(subStrides[0], it[0], body)
    when idx == 0:
      (subStrides,)
    else:
      concat(suffix_scanIt_recurse(idx - 1, t, newState, body), (subStrides,))
  else:
    block:
      let it {.inject.} = t[idx]
      let acc {.inject.} = state
      let newState = body
      when idx == 0:
        (acc,)
      else:
        concat(suffix_scanIt_recurse(idx - 1, t, newState, body), (acc,))

template prefix_scanIt*(t: untyped; startingAcc: auto; body: untyped): untyped =
  ## Left-to-right prefix scan. Injects `acc`, `it`; body → new accumulator.
  when t is int or t is Int:
    startingAcc
  else:
    prefix_scanIt_recurse(0, t, startingAcc, body)

template suffix_scanIt*(t: untyped; startingAcc: auto; body: untyped): untyped =
  ## Right-to-left suffix scan. Injects `acc`, `it`; body → new accumulator.
  when t is int or t is Int:
    startingAcc
  else:
    suffix_scanIt_recurse(tupleLen(t) - 1, t, startingAcc, body)