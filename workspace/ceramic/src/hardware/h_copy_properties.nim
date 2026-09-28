## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.


## The cp.async atom × value-type chunk views.
## The entries are declared in h_copy_registry.nim.

import std/macros
import ./h_copy_registry

{.experimental: "dynamicBindSym".}

# ═════════════════════════════════════════════════════════════════════════
#  The cp.async atom
# ═════════════════════════════════════════════════════════════════════════

type CpAsyncAtomImpl[T; NumPacked: static int] = object
  ## One 16-byte cp.async.cg chunk, NumPacked elements of type T.

type CpAsyncAtom*[T] = CpAsyncAtomImpl[T, 16 div sizeof(T)]
  ## 16-byte cp.async chunk atom for element type T,
  ## NumPacked = 16 div sizeof(T)

template numPacked*[T; NumPacked: static int](_: typedesc[CpAsyncAtomImpl[T, NumPacked]]): int =
  ## Elements per atom chunk, the chunk stays 16 bytes for any element type
  static:
    doAssert NumPacked * sizeof(T) === 16,
      "CpAsyncAtom: the chunk must be 16 bytes (the cp.async L2::128B)"
  NumPacked

template tilerMN*[T; NumPacked: static int](_: typedesc[CpAsyncAtomImpl[T, NumPacked]]): auto =
  ## The chunk tiler, (NumPacked, 1) over NumPacked consecutive elements.
  (NumPacked, 1)
