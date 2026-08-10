## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## MMA atom records — the ceramic abstraction for matrix-multiply-accumulate
## hardware primitives (tensor cores, AMX, SIMD ukernels, scalar FMA).
##
## The atom is DATA: instruction text + operand/accumulator dtypes + fragment
## layouts as (T, V) → col-major offset maps. Crucible's GpuTypeKind stays OUT
## of this record — it is backend-agnostic. dtype → emitted C++ type mapping
## happens at emission time.
##
## Design record: D30 data-record + 4 kinds (MMA POC outline).

import ./int_tuples
import ./layouts_datatypes
import ./layouts

# ═════════════════════════════════════════════════════════════════════════
#  Dtypes and scale spine
# ═════════════════════════════════════════════════════════════════════════

type
  MmaDType* = enum
    ## Operand / accumulator / scale-factor dtypes.
    ## Ceramic's own contract (not crucible's GpuTypeKind): the atom record is
    ## backend-agnostic data; the mapping to emitted C++ types happens at
    ## emission time per backend.
    mdtF32, mdtF64,
    mdtTF32,          ## 32-bit storage (f32 width), 10-bit mantissa; carried as a
                      ## uint32 BLOB in kernel staging — deliberately NOT float32-
                      ## interpreted (the mma.sync "r" operand is the raw bit pattern)
    mdtF16, mdtBF16,  ## 16-bit, packed 2-per-u32 in registers
    mdtFP8E4M3, mdtFP8E5M2,
    mdtInt8, mdtUint8, mdtInt16, mdtInt32 ## signedness is a plain field — never derived
    ## (mdtInt32: VNNI/SDOT/i8mm accumulators, per llama.cpp block-quant)

  ScaleMode* = enum
    ## How per-block scale factors attach to the MMA.
    smNone               ## plain A·B (+C)
    smInstructionOperand ## scale fed to the MMA as an operand (sm_120 _VS: ue4m3/ue8m0)
    smSoftware           ## per-block multiply done in software (CPU paths)

  ScaleVec* = enum
    ## sm_120 _VS scale operand packing granularity.
    sv4X, sv2X, sv1X     ## per-16 / per-8 / per-32 elements

  ConversionPoint* = enum
    ## Where an integer accumulator converts to f32 × scale.
    ## Atom-level accumulation property; llama.cpp uses both perBlock and
    ## endOfK, they differ in the last ulps.
    cpPerBlock, cpEndOfK

  SimdIsa* = enum
    ## CPU SIMD instruction sets carried by bkCPU_SIMD atoms.
    siAVX2, siAVX512, siNEON, siSVE, siI8MM, siVNNI, siSDOT

# ═════════════════════════════════════════════════════════════════════════
#  MmaAtom — the record
# ═════════════════════════════════════════════════════════════════════════

type
  MmaAtomKind* = enum
    bkGPU_TensorCore ## NVIDIA mma.sync / AMD MFMA+WMMA / Intel Xe DPAS — register-resident
    bkCPU_X86_AMX    ## Intel AMX: CPU tensor core — TILECFG + tmm tile registers
    bkCPU_SIMD       ## SIMD ukernels: dpbusd/dpbssd, SDOT/i8mm, FMA
    bk_FMA           ## universal scalar FMA — the (1,1,1) atom; every multiply goes through it

  MmaOperand* = enum
    opA, opB, opC

  NoLayout* = Int[-1]
    ## Sentinel layout type for kinds that do not carry fragment layouts
    ## (SIMD, FMA): they instantiate `MmaAtom[NoLayout, NoLayout, NoLayout]`.
    ## Calling a layout-derived accessor (threadCount, valuesPerThread) on
    ## such an atom fails to compile — a SIMD atom has no fragments.

  ## NOTE: generic params are intentionally unconstrained. A concept
  ## constraint (`LA: AnyLayout`) hits a Nim compiler limitation: it accepts
  ## several identical layout types but rejects three DISTINCT layout types
  ## (e.g. the AMX atoms, whose three operand layouts differ). The params
  ## are left unconstrained — layout misuse fails at the layout-algebra
  ## call sites anyway.
  MmaAtom*[
      LA = NoLayout,
      LB = NoLayout,
      LC = NoLayout] = object
    ## One MMA hardware primitive, as data.
    ##
    ## Common spine — present for every kind.
    name*: string
    mnk*: tuple[m, n, k: int]              ## M, N, K in elements
    aType*, bType*, cType*: MmaDType       ## cType is the ACCUMULATOR type
    scaleMode*: ScaleMode
    blockSize*: int                        ## per-block scale granularity along K (0 when smNone)
    sfaType*, sfbType*: MmaDType           ## scale-factor dtypes; smNone → unused

    case kind*: MmaAtomKind
    of bkGPU_TensorCore, bkCPU_X86_AMX:
      ## AMX shares this payload shape: T=1 layouts, same (T,V) → offset model.
      ## The kind discriminates MEMORY CLASS, which is the emission branch:
      ## GPU stages operands through {.shared.} smem, AMX loads tile registers
      ## from plain memory, SIMD reads cache. TILECFG + tmm assignment are
      ## emission-derivable from mnk + dtypes + layouts.
      instr*: string                       ## mma.sync… / v_mfma… / dpas / tdpbf16ps / tdpbssd
      aLayout*: LA                         ## (T, V) → col-major offset in (M, K)
      bLayout*: LB                         ## (T, V) → col-major offset in (N, K)
      cLayout*: LC                         ## (T, V) → col-major offset in (M, N)
      scaleVec*: ScaleVec                  ## GPU-only operand packing; AMX sets smNone/smSoftware
    of bkCPU_SIMD:
      isa*: SimdIsa
      nbScalars*: int                      ## scalar count per vector register row —
                                        ## semantics vary per ISA (AVX512/VNNI: 16 =
                                        ## one zmm; NEON: 8/16 = one row of i32).
                                        ## Prepared for the SIMD microkernel emitter;
                                        ## not derived from mnk.
      nbVecsNr*: int                       ## vector registers per thread along N = mnk.n div
                                        ## lanes (lanes: NEON=4, AVX512/VNNI=16) — checked
                                        ## by static asserts in the atom files
      conversionPoint*: ConversionPoint
    of bk_FMA:
      discard                              ## no payload — universal (1,1,1) atom

  TiledMma*[A: MmaAtom, TL: Layout] = object
    ## The seam: atom + thread tiling.
    ## Everything above (partition, fragments, loop) derives from this record.
    atom*: A
    threadLayout*: TL                      ## (ThrM, ThrN, ThrK) — atoms tiled across threads

# ═════════════════════════════════════════════════════════════════════════
#  Derived metadata (layout queries — no stored fields behind these)
# ═════════════════════════════════════════════════════════════════════════

func threadCount*[LA, LB, LC](atom: MmaAtom[LA, LB, LC]; operand: static MmaOperand): auto {.inline.} =
  ## Number of threads cooperating on the atom (the T-mode size of the
  ## fragment layout). All three operand layouts share the same T.
  ## COMPILE-TIME: Int[N] for static layouts (CuTe: size(ThrID{}) → Int<32>);
  ## callers needing a runtime int convert with toIntVal() explicitly.
  ## (fold/flatten on the shape tuple avoids the makeIntTuple macro quirk
  ## that `mode`/`size` hit on generic-typed const fields.)
  mixin fold, flatten
  when operand == opA: fold(flatten(atom.aLayout.shape[0]), Int[1](), acc * it)
  elif operand == opB: fold(flatten(atom.bLayout.shape[0]), Int[1](), acc * it)
  else:                fold(flatten(atom.cLayout.shape[0]), Int[1](), acc * it)

func valuesPerThread*[LA, LB, LC](atom: MmaAtom[LA, LB, LC]; operand: static MmaOperand): auto {.inline.} =
  ## Number of elements of one operand this thread holds in registers:
  ## the V-mode size of the (T, V) layout — tensor-layouts' "values per
  ## thread" (value_id in tile_mma_grid); CuTe calls the same mode FrgV.
  ## COMPILE-TIME: Int[N] for static layouts (CuTe: the V-mode of the
  ## fragment layout, checked against the arch op's register-array extent
  ## by CUTE_STATIC_ASSERT_V — mma_traits.hpp:141-144).
  ## Derives from the LAYOUTS (which must tile the operand exactly), not
  ## from mnk — the (T, V) layout is the source of truth.
  when operand == opA: cosize(atom.aLayout) div atom.threadCount(opA)
  elif operand == opB: cosize(atom.bLayout) div atom.threadCount(opB)
  else:                cosize(atom.cLayout) div atom.threadCount(opC)
