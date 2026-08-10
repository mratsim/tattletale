## make_fragment_like / make_fragment_A/B/C — layout-level tests
##
## Fragment construction (make_fragment_like/A/B) pins the V modes to the
## atom's register enumeration regardless of the operand's strides
## (make_layout_like
## compacts by stride value across all modes, so a row-major operand would
## reorder V away from the hardware order — a1/a2 swap). make_fragment_like
## flattens the V leaves to (VA,):(1|0,) and compacts the rest modes by the
## view's order, scaled after the V registers.
##
## SECTION 1 — make_fragment_like: V pinned, rest compact by value
## SECTION 2 — broadcast V kept (stride-0, full shape)
## SECTION 3 — make_layout_like divergence (V reordered on row-major inputs)
## SECTION 4 — make_fragment_A/B register order + copyFrom coordinate pin
## SECTION 5 — rank-1 passthrough

import std/assertions
import ../src/int_tuples
import ../src/layouts
import ../src/layout_constructors
import ../src/layout_indexing
import ../src/layout_algebra
import ../src/tensors
import ../src/atoms
import ../src/atoms_mma_partitioning
import ../src/kernel_gemm/atoms_nvidia
import ../src/kernel_copy_gpu

const atom = SM80_16x8x8_F32TF32TF32F32_TN
  ## (T32,V4) → (M16,K8) tf32 A fragment; V shape (2,2) → 2 V leaves.

# ── SECTION 1: V pinned, rest compact by value ─────────────────────────────
proc sec1() =
  # col-major tAv (V0, V1, RestM, RestK) — V strides follow the atom (compact)
  let colLayout = make_layout((2, 2, 1, 2), (1, 2, 4, 8))
  let colFrag = make_fragment_like(colLayout, atom.aLayout.shape[1])
  # V flattened to (4,):(1,); rest (1,2) compact by value (1,1) × V cosize 4
  doAssert colFrag === make_layout((4, 1, 2), (1, 4, 4))

  # row-major tAv — V strides stay the atom's (8,64), only the tile/rest
  # strides follow the operand (RestK stride 1 = the make_layout_like trigger)
  let rowLayout = make_layout((2, 2, 1, 2), (8, 64, 2, 1))
  let rowFrag = make_fragment_like(rowLayout, atom.aLayout.shape[1])
  # V pinned: identical V block (4,):(1,) — rest reordered by row-major values
  doAssert rowFrag === make_layout((4, 1, 2), (1, 8, 4))
  doAssert mode(rowFrag, 0).stride === 1

  # shape preserved (element count) — fragment is coordinate-compatible
  doAssert size(colFrag) === size(colLayout)
  doAssert size(rowFrag) === size(rowLayout)

# ── SECTION 2: broadcast V kept ────────────────────────────────────────────
proc sec2() =
  # V all stride-0 (broadcast) — flattened to (4,):(0,), rest compact
  let bcastLayout = make_layout((2, 2, 2), (0, 0, 8))
  let bcastFrag = make_fragment_like(bcastLayout, atom.aLayout.shape[1])
  doAssert bcastFrag === make_layout((4, 2), (0, 1))
  # broadcast V in a fragment stays broadcast (all 4 logical elements → reg 0)
  let bcastFrag2 = make_fragment_like(make_layout((2, 2, 1, 2), (0, 0, 4, 8)), atom.aLayout.shape[1])
  doAssert mode(bcastFrag2, 0).stride === 0

# ── SECTION 3: make_layout_like divergence ────────────────────────────────
proc sec3() =
  # row-major 3-mode (V, M, K):(K·M, K, 1) — V outermost, rest row-major
  let rm3 = make_layout((2, 2, 2), (4, 2, 1))
  let rmFrag = make_fragment_like(rm3, rm3.shape[0])
  let rmLike = make_layout_like(rm3)
  # fragment pins V to stride-1; make_layout_like reorders V after the
  # fast rest mode (stride 4) — the a1/a2 register scramble
  doAssert mode(rmFrag, 0).stride === 1
  doAssert mode(rmLike, 0).stride === 4
  # the two must disagree on V's order for row-major inputs
  doAssert not (rmFrag === rmLike)

# ── SECTION 4: make_fragment_A/B + coordinate copyFrom pin ─────────────────
proc sec4() =
  # partition_A-shaped view (V0, V1, RestM, RestK)
  var Aarr: array[4 * 1 * 2, uint32]
  let fakeAv = make_view(Aarr, make_layout((2, 2, 1, 2), (1, 2, 4, 8)))
  let aFrag = make_fragment_A(atom, fakeAv)
  doAssert aFrag.layout === make_layout((4, 1, 2), (1, 4, 4))
  doAssert mode(aFrag.layout, 0).stride === 1
  doAssert size(aFrag) === size(fakeAv)

  # register-order pin: fill the partition view with a row-major A tile's
  # values and check copyFrom lands them in atom register order.
  # A[m·8+k] with A row-major (K=8, 1): element (m,k) = m*8 + k.
  # V enumeration for m16n8k8 tf32: (V0,V1) = (m/8, k/4) → regs
  # (0,0),(8,0),(0,4),(8,4) — flat v = v0 + 2·v1.
  var src: array[16 * 8, uint32]
  for m in 0 ..< 16:
    for k in 0 ..< 8:
      src[m * 8 + k] = uint32(m * 8 + k)   # A[m,k] with row-major strides
  let rowA = make_view(src, make_layout((16, 8), (8, 1)))
  let tAvRm = make_fragment_like(make_layout((2, 2, 1, 2), (8, 64, 2, 1)), atom.aLayout.shape[1])
  # direct gather, mirroring gemm_tiled's fragment write:
  #   (v0, v1, 0, s) against the (V·, RestM, RestK) view
  var frag: array[4 * 1 * 2, uint32]
  let fragView = make_view(frag, tAvRm)
  for s in 0 ..< 2:
    for v in 0 ..< 4:
      let (v0, v1) = idx2crd(atom.aLayout.shape[1], v)
      let k = s * 8        # slice s of K=8 depth → k offset 0 or 8
      fragView(v, 0, s) = rowA(v0 * 8, v1 * 4 + k)   # (m, k) = (v0·8, v1·4 + k)
  # register order = V enumeration: frag.data[v + 4·s] holds A(v0·8, v1·4 + s·8)
  doAssert frag[0] == uint32(0 * 8 + 0)      # v=0: (0, 0)
  doAssert frag[1] == uint32(8 * 8 + 0)      # v=1: (8, 0) — a1
  doAssert frag[2] == uint32(0 * 8 + 4)      # v=2: (0, 4) — a2
  doAssert frag[3] == uint32(8 * 8 + 4)      # v=3: (8, 4) — a3
  doAssert frag[4] == uint32(0 * 8 + 8)      # s=1, v=0: (0, 8)
  doAssert frag[7] == uint32(8 * 8 + 12)     # s=1, v=3: (8, 12)

  # coordinate-copyFrom pin: dst(i)/src(i) decode i through their own
  # shapes — the fragment (V flat) and a differently-laid src agree
  # element-wise even though their layouts differ (NOT linear memory
  # alignment). src: (2,4):(4,1) row-major over src[] = {0..7}.
  var srcC: array[8, uint32]
  for i in 0 ..< 8: srcC[i] = uint32(i)
  var dst: array[4 * 1 * 2, uint32]
  var dstView = make_view(dst, tAvRm)
  let srcView = make_view(srcC, make_layout((2, 4), (4, 1)))
  dstView.copyFrom(srcView)
  # copyFrom iterates i over dst's size; dst(i) = src(i), i decoded through
  # each layout. dst (4,1,2):(1,4,4) → i = v + 4s; src (2,4):(4,1) →
  # (m,k) = (i mod 2, i div 2).
  doAssert dst[0] == uint32(0)              # i=0: dst (v=0,s=0) = src (0,0) → srcC[0]
  doAssert dst[3] == uint32(5)              # i=3: dst (v=3,s=0) = src (1,1) → srcC[1·4+1]=5
  doAssert dst[4] == uint32(2)              # i=4: dst (v=0,s=1) = src (0,2) → srcC[2]
  doAssert dst[7] == uint32(7)              # i=7: dst (v=3,s=1) = src (1,3) → srcC[1·4+3]=7

# ── SECTION 4b: make_fragment_B/C shapes ───────────────────────────────────
proc sec4b() =
  # B: V = (2,) single leaf — fragment (2, 1, 2):(1, 2, 2)? B rest = (1, 2)
  var Barr: array[2 * 1 * 2, uint32]
  let fakeBv = make_view(Barr, make_layout((2, 1, 2), (1, 2, 2)))
  let bFrag = make_fragment_B(atom, fakeBv)
  doAssert mode(bFrag.layout, 0).stride === 1
  # C: accumulator fragment from a partition_C-shaped view
  var Carr: array[4 * 1 * 1, float32]
  let fakeCv = make_view(Carr, make_layout((2, 2, 1, 1), (1, 2, 1, 1)))
  let cFrag = make_fragment_C(atom, fakeCv)
  doAssert mode(cFrag.layout, 0).stride === 1

  # nested-V vShape (4 leaves ((2,2),(2,2))) — flattened to (16,):(1,)
  const nestedV = ((2, 2), (2, 2))
  let nestedLayout = make_layout((2, 2, 2, 2, 1, 2), (1, 2, 4, 8, 16, 32))
  let nestedFrag = make_fragment_like(nestedLayout, nestedV)
  doAssert mode(nestedFrag, 0).stride === 1
  doAssert size(nestedFrag) === size(nestedLayout)

# ── SECTION 5: rank-1 passthrough ──────────────────────────────────────────
proc sec5() =
  # compact rank-1 passthrough
  let r1 = make_fragment_like(make_layout(Int[4]()), Int[4])
  doAssert r1 === make_layout(Int[4]())
  # broadcast rank-1 keeps stride-0 (SPEC-004)
  let r1b = make_fragment_like(make_layout(Int[4](), Int[0]()), Int[4])
  doAssert mode(r1b, 0).stride === 0
  doAssert r1b === make_layout(Int[4](), Int[0]())

# ── SECTION 5b: compile-time rejections (TEST-005) ─────────────────────────
proc sec5b() =
  # shape/stride rank mismatch
  doAssert not compiles(make_fragment_like(make_layout((2, 2, 2), (1, 2)), Int[2]))
  # V leaf count exceeds layout rank
  const threeV = (2, 2, 2)
  doAssert not compiles(make_fragment_like(make_layout((2, 2), (1, 2)), threeV))
  # mixed broadcast/non-broadcast V leaves
  const twoV = (2, 2)
  doAssert not compiles(make_fragment_like(make_layout((2, 2, 2), (0, 2, 8)), twoV))

sec1()
sec2()
sec3()
sec4()
sec4b()
sec5()
sec5b()

echo "OK: make_fragment_like / make_fragment_A/B tests passed"
