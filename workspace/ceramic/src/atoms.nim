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
## Design record: SCRATCHPAD §3l (D30 data-record + 4 kinds), POC outline
## `.SPEC_DESIGN/mma-reduced-scope-20260808.md`, lab notebook `MMA_LOG.md`.

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
    mdtTF32,          ## f32 storage, 10-bit mantissa; input of mma.sync tf32
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
    ## endOfK, they differ in the last ulps (.ANALYSIS/08-llamacpp.md:271,276,293).
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
    bkCPU_SIMD       ## SIMD ukernels (laser-style): dpbusd/dpbssd, SDOT/i8mm, FMA
    bk_FMA           ## universal scalar FMA — the (1,1,1) atom; every multiply goes through it

  MmaOperand* = enum
    opA, opB, opC

  DefaultLayout2* = Layout[(int, int), (int, int)]
    ## Default layout type for kinds that do not carry fragment layouts
    ## (SIMD, FMA). Unused fields — SIMD/FMA atoms instantiate
    ## `MmaAtom[DefaultLayout2, DefaultLayout2, DefaultLayout2]`.

  ## NOTE: generic params are intentionally unconstrained. A concept
  ## constraint (`LA: AnyLayout`) hits a Nim compiler limitation: it accepts
  ## several identical layout types but rejects three DISTINCT layout types
  ## (verified 2026-08-08, MMA_LOG entry 9). Layout misuse fails at the
  ## layout-algebra call sites anyway.
  MmaAtom*[
      LA = DefaultLayout2,
      LB = DefaultLayout2,
      LC = DefaultLayout2] = object
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
      nbScalars*: int                      ## scalar registers per thread (laser: mr×nr for f32)
      nbVecsNr*: int                       ## vector registers per thread along N
      conversionPoint*: ConversionPoint
    of bk_FMA:
      discard                              ## no payload — universal (1,1,1) atom

  TiledMma*[A, TL] = object
    ## The seam (12-cutlass-layers.md §4): atom + thread tiling.
    ## Everything above (partition, fragments, loop) derives from this record.
    atom*: A
    threadLayout*: TL                      ## (ThrM, ThrN, ThrK) — atoms tiled across threads

# ═════════════════════════════════════════════════════════════════════════
#  Accessors
# ═════════════════════════════════════════════════════════════════════════

func m*(atom: MmaAtom): int = atom.mnk.m
func n*(atom: MmaAtom): int = atom.mnk.n
func k*(atom: MmaAtom): int = atom.mnk.k

func threadCount*[LA, LB, LC](atom: MmaAtom[LA, LB, LC]; operand: MmaOperand): int =
  ## Number of threads cooperating on the atom (the T-mode size of the
  ## fragment layout). All three operand layouts share the same T.
  ## (fold/flatten on the shape tuple avoids the makeIntTuple macro quirk
  ## that `mode`/`size` hit on generic-typed const fields — MMA_LOG entry 9.)
  let tShape = case operand
               of opA: atom.aLayout.shape[0]
               of opB: atom.bLayout.shape[0]
               of opC: atom.cLayout.shape[0]
  mixin fold, flatten
  fold(flatten(tShape), Int[1](), acc * it).toIntVal()

func fragmentValsPerThread*[LA, LB, LC](atom: MmaAtom[LA, LB, LC]; operand: MmaOperand): int =
  ## Number of elements of one operand this thread holds in registers:
  ## tile size / thread count, per the (T, V) layouts.
  ## Operands: A: (M·K)/T, B: (N·K)/T, C: (M·N)/T.
  let
    m = atom.mnk.m
    n = atom.mnk.n
    k = atom.mnk.k
  case operand
  of opA: (m * k) div atom.threadCount(opA)
  of opB: (n * k) div atom.threadCount(opB)
  of opC: (m * n) div atom.threadCount(opC)
