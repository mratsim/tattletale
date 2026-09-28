## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.


## Copy-atom catalog accessors and the atom × value-type chunk views.
## The entries are declared in h_copy_registry.nim.

import std/macros
import ./h_copy_registry

{.experimental: "dynamicBindSym".}

# ═════════════════════════════════════════════════════════════════════════
#  Catalog accessors, one getter per registry property
# ═════════════════════════════════════════════════════════════════════════

macro kindOf*(atom: static CopyAtom): untyped =
  ## The atom's slot kind (ckCopy / ckCommit / ckWait).
  result = bindSym($atom & "_kind")

macro srcSpaceOf*(atom: static CopyAtom): untyped =
  ## The memory space the atom copies from, spAny when unconstrained.
  result = bindSym($atom & "_srcSpace")

macro dstSpaceOf*(atom: static CopyAtom): untyped =
  ## The memory space the atom copies into, spAny when unconstrained.
  result = bindSym($atom & "_dstSpace")

macro vecBytesOf*(atom: static CopyAtom): untyped =
  ## Bytes the atom moves per instruction, the chunk is 4, 8 or 16 wide.
  result = bindSym($atom & "_vecBytes")

macro minAlignOf*(atom: static CopyAtom): untyped =
  ## Pointer byte alignment the atom's instruction requires.
  result = bindSym($atom & "_minAlign")

macro zeroFillOf*(atom: static CopyAtom): untyped =
  ## Whether the atom supports a chunk zero-fill (the cp.async src-size-0 form).
  result = bindSym($atom & "_zeroFill")

macro cacheOf*(atom: static CopyAtom): untyped =
  ## The atom's cache behavior (cache_default / cache_always / cache_global_bypass_L1).
  result = bindSym($atom & "_cache")

macro transposeOf*(atom: static CopyAtom): untyped =
  ## Whether the atom copies with a transpose.
  result = bindSym($atom & "_transpose")

macro minCudaArchOf*(atom: static CopyAtom): untyped =
  ## Minimum CUDA architecture the atom requires, 0 = universal.
  result = bindSym($atom & "_minCudaArch")

macro instrOf*(atom: static CopyAtom): untyped =
  ## The atom's instruction spelling, "" for plain loads/stores.
  result = bindSym($atom & "_instr")

macro waitDepthOf*(atom: static CopyAtom): untyped =
  ## The wait atom's supported group depth, 0 for everything else.
  result = bindSym($atom & "_waitDepth")

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
