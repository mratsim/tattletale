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
##
## The public `crd2idx(layout, coord)` and `layout()`/`idx2crd` wrappers
## live in `layouts.nim` (they import this file).

import std/[macros, typetraits]
import ./int_tuples
import ./macros/static_for

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

template crd2idx*[Sh, St: tuple](coord: tuple; shape: Sh; stride: St): auto =
  ## Inner product: sum coord[i] * stride[i]
  ## X markers contribute 0 via operator overloads.
  ## makeIntTuple wraps static ints as Int[V] for compile-time constant folding.
  block:
    evalOnceAs(P, makeIntTuple(coord))
    evalOnceAs(D, makeIntTuple(stride))
    foldZipWith(P(), D(), Int[0]()):
      acc + it_a * it_b

template foldDim*(co, sh, st: typed; i: static int): auto =
  when i == rank(sh) - 1:
    co * st[i]
  else:
    (co mod sh[i]) * st[i] + foldDim(co div sh[i], sh, st, i + 1)

template crd2idx*[C: int or Int; Sh, St: tuple](coord: C; shape: Sh; stride: St): auto =
  ## Decompose coord across shape modes with strides.
  block:
    evalOnceAs(S, flatten(makeIntTuple(shape)))
    evalOnceAs(D, flatten(makeIntTuple(stride)))
    evalOnceAs(P, makeIntTuple(coord))
    const R = int rank(S)
    when S is tuple and R > 1:
      foldDim(P, S, D, 0)
    else:
      # Single mode after flatten — no decomposition needed
      P * D
