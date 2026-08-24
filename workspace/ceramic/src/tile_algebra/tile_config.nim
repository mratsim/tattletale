## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#        Tile config: the base atom, one source
#
# ############################################################
#
# The single hardware source of the tile layer: the per-backend,
# per-element-type atom mapping (getTileConfig), the FMA thread layout,
# the per-lane register fragment type (FragmentOf), the per-lane lane
# forms (fm/fn), and the FMA register ops over the per-lane value
# arrays. Everything else in the tile layer derives from these: the tile
# types read the atom's mnk for their subtile grid, the loads/stores/mma
# call the register ops, and no backend type name leaks past this module.
#
# The universal FMA atom is the 8×8×8 software mma: its A and C fragments
# share the Apple AC layout, so C-role == A-role and an accumulator feeds
# the next mma's A operand with zero movement. The per-atom `mma` below is
# that atom's cross-lane shuffle reduction; `zero` and `mma` are the only
# register ops the tile layer uses.

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../layout_indexing
import ../atoms
import ../kernel_gemm/atoms_universal
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  getTileConfig: the per-backend, per-element-type atom mapping
# ═════════════════════════════════════════════════════════════════════════

func getTileConfig*(T: typedesc): auto =
  ## Per-element-type atom for the enclosing DSL block's backend.
  ## fp16/bf16 widen to the fp32 accumulator. fp32 keeps its own type.
  ##
  ## Call it only inside a `metal:` / `cuda:` / `opencl:` / `vulkan:` /
  ## `webgpu:` block, or in templates and generic procs instantiated from one.
  ## Outside a DSL block the call fails with a compile-time error.
  when ccGetBackend() == ctMetal:
    when T is float16: UNIVERSAL_FMA_F16
    elif T is bfloat16: UNIVERSAL_FMA_BF16
    elif T is float32: UNIVERSAL_FMA_F32
    else: {.error: "getTileConfig: no atom for " & $T & " (fp16/bf16/fp32 only)".}
  elif ccGetBackend() == ctCuda:
    when T is float16: UNIVERSAL_FMA_F16
    elif T is bfloat16: UNIVERSAL_FMA_BF16
    elif T is float32: UNIVERSAL_FMA_F32
    else: {.error: "getTileConfig: no atom for " & $T & " (fp16/bf16/fp32 only)".}
  else:
    when T is float16: UNIVERSAL_FMA_F16
    elif T is bfloat16: UNIVERSAL_FMA_BF16
    elif T is float32: UNIVERSAL_FMA_F32
    else: {.error: "getTileConfig: no atom for " & $T & " (fp16/bf16/fp32 only)".}

# ═════════════════════════════════════════════════════════════════════════
#  The FMA thread layout: the TiledMma for the ops partition
# ═════════════════════════════════════════════════════════════════════════
#
#  The tile types carry the layout as a defaulted static param (tiles.nim),
#  so kernels never spell it. The ops build the TiledMma from the atom
#  plus this layout and cut it per thread with the atoms_mma_partitioning
#  slice coordinates.

type ThreadLayout* = object
  ## TiledMma thread layout carried by the tile types: replication counts
  ## along M, N, K. A plain record, not the generic Layout type.
  # The plain record keeps the defaulted static param unifiable under partial application.
  thrM*, thrN*, thrK*: int

const FmaThreadLayout* = ThreadLayout(thrM: 1, thrN: 1, thrK: 1)
  ## FMA config's default thread layout: one 8×8×8 atom per 32-lane
  ## threadgroup. Every lane owns its fragment slice of every subtile,
  ## and the mma's shuffle reduction is the whole atom's cross-lane work.

# ═════════════════════════════════════════════════════════════════════════
#  FragmentOf: the per-lane register fragment type
# ═════════════════════════════════════════════════════════════════════════
#
#  One atom subtile's per-lane register fragment: the per-lane value
#  array, valuesPerThread(opA) slots per thread. The 8×8×8 FMA atoms give
#  two slots per lane (the lane's (fm, fn) and (fm, fn+1) cells). The tile
#  layer reaches the per-lane elements through the register ops, never
#  through the backend type directly.

type
  FragmentOf*[A: static MmaAtom; T] = object
    ## One atom subtile's per-lane register fragment: one slot per value
    ## the atom assigns to a lane. The 8×8×8 FMA atoms give two slots per
    ## lane.
    frag*: array[toIntVal(A.valuesPerThread(opA)), T]

# ═════════════════════════════════════════════════════════════════════════
#  fm/fn: the lane forms, single-source from the atom's fragment layout
# ═════════════════════════════════════════════════════════════════════════
#
#  The per-lane (fm, fn) offsets of the A/C fragment within one atom subtile.
#  They derive from the atom's aLayout via crd2idx, never from a backend layout const.
#  For each lane bit b, the offset of the unit lane 2^b splits into the fragment row and col:
#
#    offset(2^b) = fm·A.mnk.m + fn   for the unit lane 2^b (bit b set)
#    fm = off mod A.mnk.m,  fn = off div A.mnk.m
#
#  The 8×8×8 universal and Apple atoms share the AC layout and derive the
#  five T-mode bits; the loads, the mma and the reduction tree all read
#  the lane forms from here.

func fragLaneCoeffs*[A: static MmaAtom](): (array[5, int], array[5, int]) =
  ## The (fm, fn) bit coefficients of the atom's A/C fragment layout.
  ## For lane bit b, the aLayout offset of the unit lane 2^b splits into
  ## the fragment row (fm = off mod A.mnk.m) and col (fn = off div A.mnk.m).
  ## Single source of every lane form.
  for b in 0 ..< 5:
    let off = crd2idx(A.aLayout, 1 shl b).toIntVal()
    result[0][b] = off mod A.mnk.m     # fm contribution
    result[1][b] = off div A.mnk.m     # fn contribution

func laneForm(lane: int; c0, c1, c2, c3, c4: static int): int {.inline.} =
  ## Σ_b c_b·((lane shr b) and 1): the layout's lane-bit decomposition
  ## of one lane form.
  # The coefficients are compile-time constants, so the emitted code is
  # plain bit ops. A runtime coefficient array lowers to all-zeros on the
  # Metal backend.
  (when c0 == 0: 0 else: c0 * ((lane shr 0) and 1)) +
  (when c1 == 0: 0 else: c1 * ((lane shr 1) and 1)) +
  (when c2 == 0: 0 else: c2 * ((lane shr 2) and 1)) +
  (when c3 == 0: 0 else: c3 * ((lane shr 3) and 1)) +
  (when c4 == 0: 0 else: c4 * ((lane shr 4) and 1))

func laneFm*[A: static MmaAtom](lane: int): int =
  ## The lane's fragment row within one atom subtile, from
  ## the aLayout lane-bit coefficients.
  const c = fragLaneCoeffs[A]()[0]
  laneForm(lane, c[0], c[1], c[2], c[3], c[4])

func laneFn*[A: static MmaAtom](lane: int): int =
  ## The lane's fragment col within one atom subtile
  ## (0, 2, 4, 6 on the Apple atoms), from the aLayout lane-bit
  ## coefficients.
  const c = fragLaneCoeffs[A]()[1]
  laneForm(lane, c[0], c[1], c[2], c[3], c[4])

# ═════════════════════════════════════════════════════════════════════════
#  The FMA register ops: plain per-lane arithmetic over the value arrays
# ═════════════════════════════════════════════════════════════════════════
#
#  The bk_FMA fragment is `array[valuesPerThread(opA), T]` (FragmentOf):
#  two values per lane on the 8×8×8 atom. `zero` fills the accumulator
#  seed; `mma` is the per-atom cross-lane shuffle reduction (device-only:
#  it reads the lane id and gathers the other lanes' registers).

proc zero*[N: static int; T](frag: var array[N, T]) {.inline.} =
  ## Fills the per-lane value array with 0: the accumulator seed of an in-place mma chain.
  for i in 0 ..< N:
    frag[i] = T(0)

proc mma*[A: static MmaAtom; TD; TA; TB](
    d: var array[2, TD]; a: array[2, TA]; b: array[2, TB]) {.inline.} =
  ## One 8×8×8 FMA atom's cross-lane reduction: D = A·B + D.
  ##
  ## Each lane holds A(fm, fn), A(fm, fn+1) and B(fm, fn), B(fm, fn+1)
  ## for its (fm, fn). The products it accumulates need A's k columns
  ## and B's k rows, which live in other lanes. Products accumulate in
  ## k order 0,1,2,… exactly like the CPU reference, which keeps the
  ## gemm bit-exact.
  ##
  ## Expected input: fragments of the 8×8×8 atoms. A's and D's atoms
  ## share the AC layout, B uses the B layout. The universal atoms
  ## satisfy this by construction. A is the atom whose aLayout gives
  ## the lane forms fm/fn.
  # Four k-pair steps (j = 0..3, k = 2j, 2j+1) gather the six operands
  # each step needs, first the k = 2j terms, then the k = 2j+1 terms.
  # The shuffle sources are pure bit ops on the lane id:
  #   srcA  = 2·fm + 8·(fm div 4) + (j and 1) + (j shr 1)·8
  #   srcB0 = fnBase + 4j + 8·(j div 2)
  #   srcB1 = fnBase + 4j + 2 + 8·((2j+1) div 4)
  #   fnBase = (lane and 1) + 8·((lane shr 3) and 1)
  # The 8·(fm div 4) term is the row-group wrap: rows 4–7 live in lanes
  # 16–31, not 8–15.
  let lane = int(thread_index_in_threadgroup)
  let fm = laneFm[A](lane)
  let fn = laneFn[A](lane)
  let fnBase = uint32((lane and 1) + 8 * ((lane shr 3) and 1))
  let srcABase = uint32(2 * fm + 8 * (fm div 4))
  for j in 0 ..< 4:
    let srcA = srcABase + uint32((j and 1) + (j shr 1) * 8)
    let srcB0 = fnBase + uint32(4 * j + 8 * (j div 2))
    let srcB1 = fnBase + uint32(4 * j + 2 + 8 * ((2 * j + 1) div 4))
    let a0 = simdShuffle(a[0], srcA)
    let a1 = simdShuffle(a[1], srcA)
    let b00 = simdShuffle(b[0], srcB0)
    let b01 = simdShuffle(b[1], srcB0)
    let b10 = simdShuffle(b[0], srcB1)
    let b11 = simdShuffle(b[1], srcB1)
    d[0] = d[0] + TD(a0) * TD(b00)   # k = 2j terms
    d[1] = d[1] + TD(a0) * TD(b01)
    d[0] = d[0] + TD(a1) * TD(b10)   # k = 2j+1 terms
    d[1] = d[1] + TD(a1) * TD(b11)
