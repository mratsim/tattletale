## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## MMA atom property getters and derived atom geometry.
import std/macros
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/hardware/h_configgen
import workspace/ceramic/src/hardware/h_registry

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
  ## from their layouts (`valuesPerThread`, below).
  result = bindSym($A & "_vpt")

template threadCount*(atom: static MmaAtom; operand: static MmaOperand): untyped =
  ## Threads cooperating on the atom for operand A, B or C in the `C <- A*B + C` microkernel.
  ##
  ## Returns the atom's thread count, wrapped in `Int`.
  ## Every declared GPU atom uses the same thread count for all three operands.
  Int[atom.getThreadCount()]()

template valuesPerThread*(atom: static MmaAtom; operand: static MmaOperand): untyped =
  ## Operand values per thread. A, B and C may each differ.
  ##
  ## Returns the operand layout's cosize divided by the atom's thread count.
  ##
  ## Declared atoms' V_A/V_B/V_C
  ##
  ## | Atom class                | V_A/V_B/V_C |
  ## | ------------------------- | ----------- |
  ## | tf32                      | 4/2/4       |
  ## | f16+bf16 (k16)            | 8/4/4       |
  ## | int8+e4m3 (k32)           | 16/8/4      |
  ## | universal 8x8x8 and Apple | 2/2/2       |
  ## | 1x1x1                     | 1/1/1       |
  when operand == opA: cosize(atom.getLayoutA()) div atom.threadCount(opA)
  elif operand == opB: cosize(atom.getLayoutB()) div atom.threadCount(opB)
  else:                cosize(atom.getLayoutC()) div atom.threadCount(opC)

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
