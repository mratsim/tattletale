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

func crd2idx*(coord, shape: int): int = coord
func crd2idx*[V: static int](coord: Int[V]; shape: int): int = V
func crd2idx*(coord, shape, stride: int): int = coord * stride
func crd2idx*[V: static int](coord: Int[V]; shape, stride: int): int = V * stride
func crd2idx*[V, U: static int](coord: int; shape: Int[V]; stride: Int[U]): int = coord * toIntVal(stride)
func crd2idx*[V: static int](coord: int; shape: Int[V]; stride: int): int = coord * stride
func crd2idx*[U: static int](coord: int; shape: int; stride: Int[U]): int = coord * toIntVal(stride)

# ═══════════════════════════════════════════════════════════════
#  Tuple overloads
# ═══════════════════════════════════════════════════════════════

# 3-arg: tuple coord (int or X) × tuple stride → inner product
template crd2idx*[Sh, St: tuple](coord: typed; shape: Sh; stride: St): auto =
  ## Mixed coord: X contributes 0, int contributes coord * stride.
  ## Wrap coord and stride via makeIntTuple so compile-time ints
  ## become Int[N] and stay in the Int[V] * Int[U] (→ Int[VU])
  ## operator space.
  foldZipWith(makeIntTuple(coord), makeIntTuple(stride), Int[0]()):
    acc + (when it_a is X: Int[0]() else: it_a * it_b)

template foldDim*(co, sh, st: typed; i: static int): auto =
  when i == rank(sh) - 1:
    co * st[i]
  else:
    (co mod sh[i]) * st[i] + foldDim(co div sh[i], sh, st, i + 1)

# 3-arg: int coord → decompose across modes
template crd2idx*[C: int or Int; Sh, St: tuple](coord: C; shape: Sh; stride: St): auto =
  ## Decompose coord across shape modes with strides.
  ## Recursive expression-fold (no var) so Int[N] propagates.
  ## Wrap in makeIntTuple so plain int literals become Int[N].
  type ShType = typeof(flatten(shape))
  const R = rank(ShType)
  when ShType is tuple and R > 1:
    let fshape = flatten(makeIntTuple(shape))
    let fstride = flatten(makeIntTuple(stride))
    foldDim(makeIntTuple(coord), fshape, fstride, static(0))
  else:
    # Single mode after flatten — no decomposition needed
    let sv = flatten(makeIntTuple(stride))
    coord * v
