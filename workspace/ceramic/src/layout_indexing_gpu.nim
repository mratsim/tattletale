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
##
## ── PERFORMANCE WARNING ──
## ALL crd2idx overloads in this file MUST be `template`, NOT `func`.
##
## Reason: a non-`{.inline.}` `func` lands in a separate C++ compilation unit
## (its own .cpp file), which prevents cross-module inlining at the C++ level.
## Even `{.inline.}` generates a standalone C++ function definition that the
## C++ inliner must process — and when arguments involve `Int` types
## (which become C structs in Nim's C++ backend), the struct-wrapped
## parameters add complexity that hinders optimization.
##
## A `template` produces zero C++ function definitions. The expression
## appears directly at the call site, giving the cleanest C++ output:
## bare arithmetic like `i * 16 + j * 1` with no struct wrappers around
## the Int[V] stride values.
##
## History: commit b93bb95 ('Recover 20GFlops on layout algebra CPU GEMM')
## changed crd2idx from func→template, recovering ~12× on flat-index copies
## (6ms → 0.5ms for 524k B-packing elements). The nested `toIntVal` and
## `Int[V] × int` operators must also be templates (see int_tuples_datatypes).
## ─────────────────────────

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
  ## X markers contribute 0 via operator overloads (layout_indexing.nim).
  ## makeIntTuple wraps static ints as Int[V] for compile-time constant folding.
  ##
  ## PERF: Must stay template (not func). The `it_a * it_b` delegates to
  ## Int[V]*int operator overloads which are ALSO templates (see genBinOp
  ## in int_tuples_datatypes.nim). A func chain here would prevent C++ inlining.
  block:
    evalOnceAs(P, makeIntTuple(coord))
    evalOnceAs(D, makeIntTuple(stride))
    # Int[0]() accumulator + uniform `acc + it_a * it_b` works because
    # X*int→Int[0]() operator overloads neutralize markers.
    foldZipWith(P(), D(), Int[0]()):
      acc + it_a * it_b

template foldDim*(co, sh, st: typed; i: static int): auto =
  when i == rank(sh) - 1:
    co * st[i]
  else:
    (co mod sh[i]) * st[i] + foldDim(co div sh[i], sh, st, i + 1)

template crd2idx*[C: int or Int; Sh, St: tuple](coord: C; shape: Sh; stride: St): auto =
  ## Decompose coord across shape modes with strides.
  ##
  ## PERF: Must stay template. Uses `int rank(S)` instead of `toIntVal rank(S)`
  ## to avoid a toIntVal call (even at compile time — it's a func).
  block:
    evalOnceAs(S, flatten(makeIntTuple(shape)))
    evalOnceAs(D, flatten(makeIntTuple(stride)))
    evalOnceAs(P, makeIntTuple(coord))
    const R = int rank(S)  # int() avoids toIntVal func call
    when S is tuple and R > 1:
      foldDim(P, S, D, 0)
    else:
      # Single mode after flatten — no decomposition needed
      P * D
