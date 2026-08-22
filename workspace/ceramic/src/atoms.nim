## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## MMA Atoms -> Low-level MMA (Matrix-Multiply-Accumulate) hardware descriptors
##
## Glossary:
## - MMA:
##     Fused matrix multiply-accumulate instruction,
##     C <- A·B + C with C the accumulator.
## - Atom:
##     one MMA instruction descriptor
##     - MNK tile,
##     - operand datatypes,
##     - A mapping each thread's registers to tile elements.
## - Tile: the element block one atom computes per invocation.
##     - A is (M, K)
##     - B is (N, K)
##     - C is (M, N)
## - Fragment: the register values a thread holds for one operand tile, in the order the instruction expects.
## - T: the atom's threads.
## - V: the values each thread holds in registers.
##
## Hardware schema:
##   the m16n8k8 tf32 atom as a tiled GEMM, C = A·B.
##   Each cell is the owning (thread, value):
##   - T = the atom's threads (0..31),
##   - v = the value's index in that thread's registers.
##     v is local to the thread:
##     T00.v0 and T01.v0 are different elements in different threads.
##   The k axis aligns. A's k column pairs with B's k row.
##
##                                                                   ┌──────────────────────────────────────────────────────────────┐
##                                                                   │ B (8, 8) — rows k, columns n                                 │
##                                                                   │ k=0  T00.v0 T04.v0 T08.v0 T12.v0 T16.v0 T20.v0 T24.v0 T28.v0 │
##                                                                   │ k=1  T01.v0 T05.v0 T09.v0 T13.v0 T17.v0 T21.v0 T25.v0 T29.v0 │
##                                                                   │ k=2  T02.v0 T06.v0 T10.v0 T14.v0 T18.v0 T22.v0 T26.v0 T30.v0 │
##                                                                   │ k=3  T03.v0 T07.v0 T11.v0 T15.v0 T19.v0 T23.v0 T27.v0 T31.v0 │
##                                                                   │ k=4  T00.v1 T04.v1 T08.v1 T12.v1 T16.v1 T20.v1 T24.v1 T28.v1 │
##                                                                   │ k=5  T01.v1 T05.v1 T09.v1 T13.v1 T17.v1 T21.v1 T25.v1 T29.v1 │
##                                                                   │ k=6  T02.v1 T06.v1 T10.v1 T14.v1 T18.v1 T22.v1 T26.v1 T30.v1 │
##                                                                   │ k=7  T03.v1 T07.v1 T11.v1 T15.v1 T19.v1 T23.v1 T27.v1 T31.v1 │
##                                                                   └──────────────────────────────────────────────────────────────┘
##
## ┌───────────────────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────────────────┐
## │ A (16, 8) — rows m, columns k                                 │ │ C (16, 8) — rows m, columns n                                │
## │ m=0  T00.v0 T01.v0 T02.v0 T03.v0 T00.v2 T01.v2 T02.v2 T03.v2  │ │ m=0  T00.v0 T00.v1 T01.v0 T01.v1 T02.v0 T02.v1 T03.v0 T03.v1 │
## │ m=1  T04.v0 T05.v0 T06.v0 T07.v0 T04.v2 T05.v2 T06.v2 T07.v2  │ │ m=1  T04.v0 T04.v1 T05.v0 T05.v1 T06.v0 T06.v1 T07.v0 T07.v1 │
## │ m=2  T08.v0 T09.v0 T10.v0 T11.v0 T08.v2 T09.v2 T10.v2 T11.v2  │ │ m=2  T08.v0 T08.v1 T09.v0 T09.v1 T10.v0 T10.v1 T11.v0 T11.v1 │
## │ m=3  T12.v0 T13.v0 T14.v0 T15.v0 T12.v2 T13.v2 T14.v2 T15.v2  │ │ m=3  T12.v0 T12.v1 T13.v0 T13.v1 T14.v0 T14.v1 T15.v0 T15.v1 │
## │ m=4  T16.v0 T17.v0 T18.v0 T19.v0 T16.v2 T17.v2 T18.v2 T19.v2  │ │ m=4  T16.v0 T16.v1 T17.v0 T17.v1 T18.v0 T18.v1 T19.v0 T19.v1 │
## │ m=5  T20.v0 T21.v0 T22.v0 T23.v0 T20.v2 T21.v2 T22.v2 T23.v2  │ │ m=5  T20.v0 T20.v1 T21.v0 T21.v1 T22.v0 T22.v1 T23.v0 T23.v1 │
## │ m=6  T24.v0 T25.v0 T26.v0 T27.v0 T24.v2 T25.v2 T26.v2 T27.v2  │ │ m=6  T24.v0 T24.v1 T25.v0 T25.v1 T26.v0 T26.v1 T27.v0 T27.v1 │
## │ m=7  T28.v0 T29.v0 T30.v0 T31.v0 T28.v2 T29.v2 T30.v2 T31.v2  │ │ m=7  T28.v0 T28.v1 T29.v0 T29.v1 T30.v0 T30.v1 T31.v0 T31.v1 │
## │ m=8  T00.v1 T01.v1 T02.v1 T03.v1 T00.v3 T01.v3 T02.v3 T03.v3  │ │ m=8  T00.v2 T00.v3 T01.v2 T01.v3 T02.v2 T02.v3 T03.v2 T03.v3 │
## │ m=9  T04.v1 T05.v1 T06.v1 T07.v1 T04.v3 T05.v3 T06.v3 T07.v3  │ │ m=9  T04.v2 T04.v3 T05.v2 T05.v3 T06.v2 T06.v3 T07.v2 T07.v3 │
## │ m=10 T08.v1 T09.v1 T10.v1 T11.v1 T08.v3 T09.v3 T10.v3 T11.v3  │ │ m=10 T08.v2 T08.v3 T09.v2 T09.v3 T10.v2 T10.v3 T11.v2 T11.v3 │
## │ m=11 T12.v1 T13.v1 T14.v1 T15.v1 T12.v3 T13.v3 T14.v3 T15.v3  │ │ m=11 T12.v2 T12.v3 T13.v2 T13.v3 T14.v2 T14.v3 T15.v2 T15.v3 │
## │ m=12 T16.v1 T17.v1 T18.v1 T19.v1 T16.v3 T17.v3 T18.v3 T19.v3  │ │ m=12 T16.v2 T16.v3 T17.v2 T17.v3 T18.v2 T18.v3 T19.v2 T19.v3 │
## │ m=13 T20.v1 T21.v1 T22.v1 T23.v1 T20.v3 T21.v3 T22.v3 T23.v3  │ │ m=13 T20.v2 T20.v3 T21.v2 T21.v3 T22.v2 T22.v3 T23.v2 T23.v3 │
## │ m=14 T24.v1 T25.v1 T26.v1 T27.v1 T24.v3 T25.v3 T26.v3 T27.v3  │ │ m=14 T24.v2 T24.v3 T25.v2 T25.v3 T26.v2 T26.v3 T27.v2 T27.v3 │
## │ m=15 T28.v1 T29.v1 T30.v1 T31.v1 T28.v3 T29.v3 T30.v3 T31.v3  │ │ m=15 T28.v2 T28.v3 T29.v2 T29.v3 T30.v2 T30.v3 T31.v2 T31.v3 │
## └───────────────────────────────────────────────────────────────┘ └──────────────────────────────────────────────────────────────┘
##
## T00's register values: A v0..v3 = (m, k) (0,0) (8,0) (0,4) (8,4).
## B v0..v1 = (k, n) (0,0) (4,0).
## C v0..v3 = (m, n) (0,0) (0,1) (8,0) (8,1).
## Each C value sums A(m,k)·B(k,n) over the 8 k entries. The k terms live in
## different threads (T00 holds only k ∈ {0, 4} of A and B), and the
## instruction sums them into the owning thread's register.

import ./int_tuples
import ./layouts_datatypes
import ./layouts

# ═════════════════════════════════════════════════════════════════════════
#  Datatypes and SIMD ISAs
# ═════════════════════════════════════════════════════════════════════════

type
  MmaDType* = enum
    ## Matrix-Multiply-Accumulate (MMA) datatypes.
    mdtF32, mdtF64,
    mdtTF32,          ## specialized tensor float32, 32-bit with 10-bit mantissa in a 32-bit "opaque" blob
    mdtF16, mdtBF16,  ## 16-bit, packed 2-per-u32 in registers
    mdtFP8E4M3,       ## 8-bit: 1 sign + 4 exponent + 3 mantissa bits
    mdtFP8E5M2,       ## 8-bit: 1 sign + 5 exponent + 2 mantissa bits
    mdtInt8, mdtUint8, mdtInt16, mdtInt32

  SimdIsa* = enum
    siAVX2, siAVX512, siNEON, siSVE, siI8MM, siVNNI, siSDOT

# ═════════════════════════════════════════════════════════════════════════
#  MmaAtom — the record
# ═════════════════════════════════════════════════════════════════════════

type
  MmaAtomKind* = enum
    bkGPU_TensorCore ## NVIDIA mma.sync / AMD MFMA+WMMA / Intel Xe DPAS
    bkCPU_X86_AMX    ## Intel AMX: CPU tensor core
    bkCPU_SIMD       ## SIMD ukernels: dpbusd/dpbssd, SDOT/i8mm, FMA
    bk_FMA           ## Fallback scalar FMA

  MmaOperand* = enum
    ## Matrix operand in the standard GEMM description α·AB + β·C.
    opA, opB, opC

  NoLayout* = Int[-1]
    ## Sentinel layout.

  MmaAtom*[LA, LB, LC] = object
    ## Matrix-Multiply-Accumulate (MMA) hardware descriptor.
    name*: string
    mnk*: tuple[m, n, k: int]
      ## Tile dims in elements: A is (M, K), B is (N, K), C the (M, N) result.
      ## B is stored (N, K) as in Nvidia CUTLASS, the transpose of the textbook (K, N).
    aType*, bType*, cType*: MmaDType       ## cType is the ACCUMULATOR type
    # TODO: pending the sm_120 _VS atom and its kernel.
    case kind*: MmaAtomKind
    of bkGPU_TensorCore, bkCPU_X86_AMX, bk_FMA:
      instr*: string                       ## mma.sync… / v_mfma… / dpas / tdpbf16ps / tdpbssd;
                                           ## empty for bk_FMA — its instruction is plain arithmetic
      aLayout*: LA                         ## (T, V) → col-major offset in (M, K)
      bLayout*: LB                         ## (T, V) → col-major offset in (N, K)
      cLayout*: LC                         ## (T, V) → col-major offset in (M, N)
    of bkCPU_SIMD:
      isa*: SimdIsa
      nbScalarsPerVector*: int
      nbVectorsPerTile*: int

  TiledMma*[A: MmaAtom, TL: Layout] = object
    ## Compile-time record: the atom plus its (ThrM, ThrN, ThrK) tiling across threads.
    ## ThrM, ThrN, ThrK: the atom's replication counts along M, N, K.
    ## Covered tile = (ThrM·M, ThrN·N, ThrK·K).
    ## Total threads = atom.threadCount × ThrM·ThrN·ThrK.
    ## Example (m16n8k8 tf32 atom, tiling (2, 2, 1)):
    ##   covered tile (32, 16, 8), 128 threads
    ##   thread 32 (atom position (1, 0)) holds the A fragment
    ##   (16,0) (24,0) (16,4) (24,4).
    ## The atom is fixed by the hardware. The tiling is a software choice.
    ## Downstream consumers:
    ## - thread-fragment layouts (thrfrg_A/B/C),
    ## - operand partitions (partition_A/B/C),
    ## - the kernel's K-loop.
    atom*: A
    threadLayout*: TL                      ## (ThrM, ThrN, ThrK)


# ═════════════════════════════════════════════════════════════════════════
#  Derived metadata — layout queries, no stored fields
# ═════════════════════════════════════════════════════════════════════════

func threadCount*[LA, LB, LC](atom: MmaAtom[LA, LB, LC]; operand: static MmaOperand): auto {.inline.} =
  ## Returns the number of threads cooperating on the atom for operand matrix A, B or C in `C <- A*B + C` microkernel
  mixin fold, flatten
  when operand == opA: fold(flatten(atom.aLayout.shape[0]), Int[1](), acc * it)
  elif operand == opB: fold(flatten(atom.bLayout.shape[0]), Int[1](), acc * it)
  else:                fold(flatten(atom.cLayout.shape[0]), Int[1](), acc * it)

func valuesPerThread*[LA, LB, LC](atom: MmaAtom[LA, LB, LC]; operand: static MmaOperand): auto {.inline.} =
  when operand == opA: cosize(atom.aLayout) div atom.threadCount(opA)
  elif operand == opB: cosize(atom.bLayout) div atom.threadCount(opB)
  else:                cosize(atom.cLayout) div atom.threadCount(opC)

func thrM*(tma: static TiledMma): static int {.inline.} =
  toIntVal(tma.threadLayout.shape[0])

func thrN*(tma: static TiledMma): static int {.inline.} =
  toIntVal(tma.threadLayout.shape[1])

func thrK*(tma: static TiledMma): static int {.inline.} =
  toIntVal(tma.threadLayout.shape[2])

func threadCount*(tma: static TiledMma): static int {.inline.} =
  ## Total threads the TiledMma uses: T × ThrM·ThrN·ThrK.
  toIntVal(tma.atom.threadCount(opA)) * tma.thrM * tma.thrN * tma.thrK
