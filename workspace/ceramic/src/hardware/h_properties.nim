## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## MMA atom property getters — method-call syntax on the enum.
##
## Each getter resolves the per-atom const the registry generated
## (`NAME_m`, `NAME_aLayout`, …) via `bindSym($Name & "_suffix")`, the Constantine
## `getCoefA`/`baseFieldModulus` pattern. The resolved const
## keeps its declared type: scalars fold as ints, the layout getters
## return the real Layout consts (heterogeneous per atom — no typed
## layout table exists, and none is possible: the SM80 16×8×8 and the 1×1×1
## atoms have structurally distinct Layout types).
##
## Usage: `A.getM()`, `A.getLayoutA()` with `A: static MmaAtom`, in type
## bodies (`array[A.getVpt(), T]`), in `const` blocks and in `when`
## branches.
##
## `{.experimental: "dynamicBindSym".}` is required: without it the computed-name
## `bindSym($A & "_suffix")` fails to evaluate `A` at
## compile time.

import std/macros
import ./h_registry

{.experimental: "dynamicBindSym".}

macro getM*(A: static MmaAtom): untyped =
  ## Returns the atom's M dimension: the A tile's rows, the C tile's rows.
  result = bindSym($A & "_m")

macro getN*(A: static MmaAtom): untyped =
  ## Returns the atom's N dimension: the B tile's rows, the C tile's columns.
  result = bindSym($A & "_n")

macro getK*(A: static MmaAtom): untyped =
  ## Returns the atom's K dimension: the contraction length (A's columns, B's columns).
  result = bindSym($A & "_k")

macro getVpt*(A: static MmaAtom): untyped =
  ## Returns the A fragment's values per thread (V in the (T, V) layout),
  ## the registry's per-atom `vpt` const. The B and C fragments' V derive
  ## from their layouts (`valuesPerThread`, in atoms_mma_partitioning).
  result = bindSym($A & "_vpt")

macro getThreadCount*(A: static MmaAtom): untyped =
  ## Returns the number of threads one atom invocation cooperates over.
  result = bindSym($A & "_threadCount")

macro getLayoutA*(A: static MmaAtom): untyped =
  ## Returns the atom's A fragment layout: (T, V) → col-major offset in (M, K).
  result = bindSym($A & "_aLayout")

macro getLayoutB*(A: static MmaAtom): untyped =
  ## Returns the atom's B fragment layout: (T, V) → col-major offset in (N, K).
  result = bindSym($A & "_bLayout")

macro getLayoutC*(A: static MmaAtom): untyped =
  ## Returns the atom's C fragment layout: (T, V) → col-major offset in (M, N).
  result = bindSym($A & "_cLayout")

macro getInstr*(A: static MmaAtom): untyped =
  ## Returns the atom's instruction mnemonic: the `mma.sync…` asm string,
  ## `"simdgroup_multiply_accumulate"` for the Apple simdgroup atoms, or
  ## `""` for the universal FMA atoms (plain arithmetic, no instruction).
  ## Kind checks dispatch on this value.
  result = bindSym($A & "_instr")
