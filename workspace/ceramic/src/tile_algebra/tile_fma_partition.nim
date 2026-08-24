## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## FMA thread partition: the per-thread subtile ownership.
##
## The bk_FMA atom is the 8×8×8 software mma (T=32). The thread layout
## (1, 1, 1) keeps one atom per 32-lane threadgroup. The congruence
## degenerates so every lane owns its fragment slice of every subtile.
## The tile type carries the layout as a defaulted static parameter.
## Per thread, the threadgroup index splits into (tm, tn, tk) coordinates:
##   - the A operand: row subtiles n ≡ tm (mod ThrM), every k subtile.
##   - the B operand: col subtiles m ≡ tn (mod ThrN), every k subtile.
##   - the C/D operand: rows ≡ tm (mod ThrM), cols ≡ tn (mod ThrN).
## The register storage uses the congruence directly.
## Kernels never see any of this: the ops read the threadgroup index.

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../atoms
import ../atoms_mma_partitioning
import ./tile_config
import workspace/crucible

func fmaTma*[A: static MmaAtom; TL: static ThreadLayout](): auto =
  ## The FMA config's TiledMma: the atom plus the tile's thread layout.
  TiledMma[A, make_layout((TL.thrM, TL.thrN, TL.thrK))](
    atom: A, threadLayout: make_layout((TL.thrM, TL.thrN, TL.thrK)))

proc fmaSlice*[A: static MmaAtom; TL: static ThreadLayout](): ThrSlice =
  ## This thread's (tm, tn, tk) in the config's TiledMma.
  get_slice(fmaTma[A, TL](), int(thread_index_in_threadgroup))

proc fmaOwnsA*[A: static MmaAtom; TL: static ThreadLayout](n: int): bool =
  ## Whether this thread owns A row subtile n: n ≡ tm (mod ThrM).
  ## The k subtiles are all owned (the register-replicated k dim).
  (n mod TL.thrM) == fmaSlice[A, TL]().tm

proc fmaOwnsB*[A: static MmaAtom; TL: static ThreadLayout](m: int): bool =
  ## Whether this thread owns B col subtile m: m ≡ tn (mod ThrN).
  ## The k subtiles are all owned (the register-replicated k dim).
  (m mod TL.thrN) == fmaSlice[A, TL]().tn

proc fmaOwnsD*[A: static MmaAtom; TL: static ThreadLayout](n, m: int): bool =
  ## Whether this thread owns C/D subtile (n, m): n ≡ tm (mod ThrM),
  ## m ≡ tn (mod ThrN).
  let thr = fmaSlice[A, TL]()
  (n mod TL.thrM) == thr.tm and
  (m mod TL.thrN) == thr.tn

func fmaTree*[A: static MmaAtom; TL: static ThreadLayout](): (array[8, int], int, uint32) =
  ## Row-reduction shuffle tree: (deltas, step count, leader mask).
  ## The lanes that share a fragment row (same fm) hold that row's column
  ## pairs, so the tree walks the lane's fragment-column bits: for the
  ## 8×8×8 atom the row group is {base, base+1, base+8, base+9}, i.e. the
  ## lanes 1 and 8 apart, and the leader mask clears bits 0 and 3. The
  ## deltas are 2^b for each lane bit b whose fn coefficient is nonzero
  ## (Apple/universal AC layout: fn = 2·b0 + 4·b3 → deltas [1, 8],
  ## mask 0b01001).
  ## A one-lane atom (threadCount 1) needs no cross-lane reduction:
  ## no steps, zero mask, a pure per-lane fold.
  when toIntVal(A.threadCount(opA)) == 1:
    discard
  else:
    # Unrolled over the 5 lane bits: each nonzero fn coefficient adds
    # its bit's delta (2^b) to the tree and to the leader mask.
    const fnCoeffs = fragLaneCoeffs[A]()[1]
    var steps = 0
    when fnCoeffs[0] != 0:
      result[0][steps] = 1
      result[2] = result[2] or 1'u32
      inc steps
    when fnCoeffs[1] != 0:
      result[0][steps] = 2
      result[2] = result[2] or 2'u32
      inc steps
    when fnCoeffs[2] != 0:
      result[0][steps] = 4
      result[2] = result[2] or 4'u32
      inc steps
    when fnCoeffs[3] != 0:
      result[0][steps] = 8
      result[2] = result[2] or 8'u32
      inc steps
    when fnCoeffs[4] != 0:
      result[0][steps] = 16
      result[2] = result[2] or 16'u32
      inc steps
    result[1] = steps
