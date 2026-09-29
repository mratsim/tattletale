## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU-suitable indexing: crd2idx (coord→idx) and idx2crd (idx→coord).
##
## These are the raw computation functions (no Layout imports).
## They operate on shape/stride tuples and scalars.

import std/[macros, typetraits]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/macros/static_for

#  Markers for slice and dice
# ═══════════════════════════════════════════════════════════════

type
  X* = object  ## slice: keep this dimension, dice: drop this dimension
  Y* = object  ## dice: keep this dimension, slice: drop this dimension

const _* = X()  ## value-level marker for free/slice dimensions

# X marker arithmetic: X contributes 0 in inner products
# X*Int[V] returns Int[0] (not plain int) so compile-time constant folding
# preserves the Int type system. X*int is plain int for runtime values.
template `*`*(c: X; s: int): Int[0] = Int[0]()
template `*`*[V: static int](c: X; s: Int[V]): Int[0] = Int[0]()
template `*`*(s: int; c: X): Int[0] = Int[0]()
template `*`*[V: static int](s: Int[V]; c: X): Int[0] = Int[0]()

template mapLeavesWith*(singleton: X, body: untyped): X =
  singleton

# ═══════════════════════════════════════════════════════════════
#  Scalar overloads
# ═══════════════════════════════════════════════════════════════

template crd2idx*(coord, shape: int): int = coord
template crd2idx*[V: static int](coord: Int[V]; shape: int): Int[V] = V
template crd2idx*(coord, shape, stride: int): int = coord * stride
template crd2idx*[V: static int](coord: Int[V]; shape, stride: int): int = coord * stride
template crd2idx*[V, U: static int](coord: int; shape: Int[V]; stride: Int[U]): auto = coord * stride
template crd2idx*[V: static int](coord: int; shape: Int[V]; stride: int): auto = coord * stride
template crd2idx*[U: static int](coord: int; shape: int; stride: Int[U]): auto = coord * stride
template crd2idx*[V, U, W: static int](coord: Int[V], shape: Int[U], stride: Int[W]): auto = coord * stride

# ═══════════════════════════════════════════════════════════════
#  Tuple overloads
# ═══════════════════════════════════════════════════════════════

template crd2idxDimension*(coord, shape, stride: typed): auto =
  ## Per-dimension crd2idx anchored on the shape dimension structure.
  when coord is X:
    # X markers contribute 0 at any nesting level
    Int[0]()
  elif shape is tuple:
    when coord is tuple:
      # Nested coord into a nested dimension: recurse over the sub-dimensions
      crd2idxRecur(coord, shape, stride, 0)
    else:
      # Scalar coord into a nested dimension: delegate to the scalar
      # decomposition path (foldDim over the dimension's leaves)
      crd2idx(coord, shape, stride)
  else:
    # Flat dimension: inner product of the coord element with the stride
    coord * stride

template crd2idxRecur*(coord, shape, stride: typed; i: static int): auto =
  ## Sum the per-dimension contributions of a tuple coord over the shape.
  when i == rank(shape) - 1:
    crd2idxDimension(coord[i], shape[i], stride[i])
  else:
    crd2idxDimension(coord[i], shape[i], stride[i]) +
      crd2idxRecur(coord, shape, stride, i + 1)

template crd2idx*[Sh, St: tuple](coord: tuple; shape: Sh; stride: St): auto =
  ## Recursive over shape: each top-level dimension is dispatched on
  ## its own (coord element, shape dimension, stride dimension).
  block:
    evalOnceAs(P, makeIntTuple(coord))
    evalOnceAs(S, makeIntTuple(shape))
    evalOnceAs(D, makeIntTuple(stride))
    crd2idxRecur(P(), S(), D(), 0)

macro foldDim*(co, sh, st: typed; i: static int): auto =
  ## Args:
  ## - `co`, the coord expression
  ## - `sh`, `st` flat tuples for shape and stride
  ## - `i`, the first component to accumulate
  ## Returns one nested expression summing per-component contributions
  ##   `(co div prior components mod sh[k]) * st[k]` over k in i .. rank-1.
  block:
    var shLeaves, stLeaves: seq[NimNode]
    for (leaf, _) in flatLeaves(sh):
      shLeaves.add leaf
    for (leaf, _) in flatLeaves(st):
      stLeaves.add leaf
    let r = shLeaves.len
    if stLeaves.len != r:
      error("foldDim: shape and stride leaf counts differ: " & $r & " vs " & $stLeaves.len)
    if r == 0:
      error("foldDim: empty shape")
    proc foldFrom(c: NimNode, k: int): NimNode =
      let shK = shLeaves[k]
      let stK = stLeaves[k]
      if k == r - 1:
        result = quote do: `c` * `stK`
      else:
        let rest = foldFrom(quote do: `c` div `shK`, k + 1)
        result = quote do: (`c` mod `shK`) * `stK` + `rest`
    result = foldFrom(co, i)

template crd2idx*[C: int or Int; Sh, St: tuple](coord: C; shape: Sh; stride: St): auto =
  ## Decompose coord across shape dimensions with strides.
  block:
    evalOnceAs(P, makeIntTuple(coord))
    foldDim(P, makeIntTuple(shape), makeIntTuple(stride), 0)
