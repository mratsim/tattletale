# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Jagged attention op: kernel, graph node and Tensor face op colocated.
##
## Same file shape as the pool write op:
##
## - the device kernel
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk by registerOp
## - the Tensor face op over the erased engine
##
## The kernel is the transformer-specialized paged attention.
## One launch per op over the PagePool slabs.
##
## Statics (refusals at the face otherwise):
##
## - page_size 256
## - head dim 128
##
## Output binding:
##
## - the node's impl is a fresh (q_total, H, D) bfloat16 cell, materialized
##   at enqueue at offset 0 and fully overwritten by the kernel
## - the handler replaces the walked node with the materialized output
##   after the enqueue
##
## Metadata:
##
## - q_offsets and the KV tables stage into int32 device buffers per eval.
##   The staged values narrow from host Nim ints, the face contracts bound
##   every staged value to the int32 domain
## - the launch closure captures only the TensorBridges and the staged
##   metadata DeviceMem. The batch's borrow tracking retains device buffers
##   at enqueue and releases them on batch destroy, so staged buffers
##   survive past submit

import std/typetraits
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/crucible/src/builtins/builtins_gpu_types
import workspace/iface
import ../../tensors_internals
import ../dispatch
import workspace/design_by_contract
import workspace/math

# Batch enqueue walks the kernel body at compile time and resolves the tile
# algebra's thread-coordinate templates in the importing module's scope.
export tile_algebra

# ############################################################
#
#                   Jagged attention
#
# ############################################################

const
  attnTile = 32
  ## QK mma schedule tile. page_size 256 is a multiple of the 32-token KV
  ## tile so each KV tile lies inside one page.
  PageSize = 256
  ## Pool page token capacity, the op's static page size.
  HeadDim = 128
  ## Static head dim of the op.

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc jagged_attn_fwd*(
    o, q, k_pool, v_pool: distinct TensorView,
    q_offsets: distinct TensorView,
    kvOffsets, kvIndices, kvSeqlens: ptr UncheckedArray[int32],
    scale: float32, layer, H, Nkv: int,
    page_size, D: static int,
    windowed: static bool = false, windowSize: int = low(int32),
    mmaQK: static TiledMma) =
  ## Launch dims: (q token blocks, H, num_seqs), 32 lanes.
  ##
  ## - x: the segment's mt-token q block
  ## - y: head
  ## - z: segment
  ##
  ## One grid point computes mt query tokens against the segment's KV
  ## tokens (its pages), causal-banded per token:
  ##
  ##             KV tokens (segment pages)
  ##              page 0    page 1    …
  ##   q tok 0    ███████   X         X
  ##   q tok 1    ███████   ██        X
  ##   …          ███████   ███████   ██
  ##   q tok mt−1 ███████   ███████   ██
  ##
  ##   █ = attended (kPos <= bandBase + qPos), X = masked
  ##
  ## - a windowed launch masks the KV tokens more than windowSize - 1 below
  ##   each query token's position bandBase + qPos, and the loop start skips
  ##   the tiles entirely below the window
  ##
  ## Dataflow, one threadgroup, per KV block of the online softmax, tensors on edges, ops in boxes:
  ##
  ##   q (mt, D) ──┐
  ##               ▼
  ##   K (D, nt) ──► [ S = q·Kᵀ (fp32) ]──× q_mul──► [ band mask ]──► [ m ← max(m, S) ]
  ##                                                                      │        │
  ##   P̃ = exp2(S − m) ◄──────────────────────────────────────────────────┘        ▼
  ##         │                                                        rescale = exp2(m_prev − m)
  ##         │            ┌─────────────────────────────────────────────────────────┐
  ##         └───────────►│  l ← l·rescale + tokensum(P̃)                            │
  ##                      │                                                         │
  ##   V (nt, D) ────────►│  O ← O·rescale + P̃·V (fp32)                             │
  ##                      └────────────────────────────┬────────────────────────────┘
  ##                                                   │  per KV block
  ##                                                   ▼
  ##                                o (mt, D) = O / l, element-dtype store
  ##
  ## Output:
  ##
  ## - o is fully written over the segment's q tokens and the banded mask gives masked KV blocks zero attention weight
  static: doAssert mmaQK.atom.getThreadCount() == getLaneWidth(),
    "jagged attention: one simdgroup per threadgroup, the atom must cover the lanes"
  static: doAssert page_size mod mmaQK.NT == 0,
    "jagged attention: a kv tile's tokens must lie inside one page"

  const
    mt = mmaQK.MT
    nt = mmaQK.NT
    qkMma = tiledMma(float32, q.T, k_pool.T, mt, nt, D)
    pvMma = tiledMma(float32, q.T, v_pool.T, mt, D, nt)
    qMap  = qkMma.getThreadValueMap(opA)       # Q
    kMap  = qkMma.getThreadValueMap(opB)       # K: (S, Dk) storage, (N, K) view
    vMap  = pvMma.getThreadValueMap(opB)       # Vᵀ: (Dv, S) storage, (N, K) view
    oMap  = pvMma.getThreadValueMap(opC)
    qkMap = qkMma.getThreadValueMap(opC)       # scores: band cut
    qkCLayout = qkMma.getMmaLayout(opC)        # S tile, per lane
    pvCLayout = pvMma.getMmaLayout(opC)        # O tile, per lane
    rowStatsL = reducedLayout(qkMap.layout, 1)

  let qMul = scale * Log2e                     # exp2 shape of the softmax

  let
    qBlock0 = threadgroup_position_in_grid.x
    (qStride, _, _) = (threadgroups_per_grid.x, threadgroups_per_grid.y, threadgroups_per_grid.z)
    head = threadgroup_position_in_grid.y
    segId = threadgroup_position_in_grid.z
    thrID = getLocalThreadID()

  let
    q0: int = q_offsets.data[segId]
    qLen = q_offsets.data[segId + 1] - q0
    kvLen: int = kvSeqlens[segId]
    # decode (q_len = 1) attends cached tokens only, prefill (q_len >= 2)
    # also attends the new tokens, one token more per q token
    decodeAdj = (if qLen == 1: 1 else: 0)
    cachedLen = kvLen - qLen + decodeAdj
    totalK = cachedLen + qLen - decodeAdj
    bandBase = cachedLen - decodeAdj
    # KV loop bounds: the segment's pages and the query tiles cap the walk
    pageBlocks = totalK.round_multiple_up(page_size) div nt
    kvHead = head div (H div Nkv)
    qBlocks = qLen.ceil_div(mt)
  # Per-head q/o tokens:
  # - [q0, q0 + qLen) of the packed (num_qo_tokens, H, D) tensors.
  # - Loads/stores past qLen are padded/dropped.
  let
    segQ = q(q0 ..< q0 + qLen, head, _)
    segO = o(q0 ..< q0 + qLen, head, _)

  var
    qReg: tile(q.T, qkMma.getMmaLayout(opA))    # (mt, D)
    qk = zeros(tile(float32, qkCLayout))        # S = Q·Kᵀ scores, (mt × nt)

  # Each workgroup starts at its own query tile and steps by the grid width,
  # so a grid smaller than the tile count still covers every tile
  for qBlock in countup(qBlock0, qBlocks - 1, qStride):
    # Per-tile softmax state: a workgroup runs several tiles when the grid
    # is smaller than the tile count
    var
      mI = finiteFloor(tile(float32, rowStatsL))  # running max over keys, (mt × 1)
        # k_tile_sdpa's finite-floor init: with bandBase >= 0 the first tile always contains the attended kPos 0
      lI = zeros(tile(float32, rowStatsL))        # running attention-weight sum, (mt × 1)
      acc = zeros(tile(float32, pvCLayout))       # (mt × D) accumulator
    let
      q0Local = qBlock * mt
      lastKey = (if cachedLen + q0Local + (mt - 1) < totalK - 1:
                   cachedLen + q0Local + (mt - 1)
                 else:
                   totalK - 1)
      lastAttBlock = lastKey div nt + 1
      kvBlocks = (if pageBlocks < lastAttBlock: pageBlocks else: lastAttBlock)
      # kvStart skips the tiles entirely below the window's left edge
      kvStart = (when windowed:
                   max(0, (bandBase + q0Local - windowSize + 1) div nt)
                 else:
                   0)
    qReg.load(segQ, qMap, thrID, (qBlock, 0))

    for kvBlk in kvStart ..< kvBlocks:
      let
        kvPos = kvBlk * nt
        pageIdx = kvPos div page_size
        inPage = kvPos - pageIdx * page_size
        pageId: int = kvIndices[kvOffsets[segId] + pageIdx]
        # Bounds at the covered tokens: a 0·NaN from the garbage tail would poison the P·V mma.
        covered = min(nt, kvLen - kvPos)
        # K subview: (covered, D) over the token stride
        # V subview: the transposed slab view (D, covered) that the P·V mma reads as Vᵀ
        vK = k_pool(pageId, layer, inPage ..< inPage + covered, kvHead, _)
        vV = v_pool(pageId, layer, inPage ..< inPage + covered, kvHead, _)
               .transpose(0, 1)

      var kReg: tile(k_pool.T, qkMma.getMmaLayout(opB))  # (nt, D)
      kReg.load(vK, kMap, thrID, (0, 0))
      qk.fillWith(0'f32)
      qkMma.atom.gemm_tile(qk, qReg, kReg)       # S = Q·Kᵀ   (MMA 1)

      qk *= qMul                                 # scale, before the band cut

      qk.applyAttentionMask(qkCLayout, qkMap, thrID, q0Local, kvPos):
        kPos > bandBase + qPos or
        (when windowed:
           kPos < bandBase + qPos - windowSize + 1
         else:
           false)

      var mIj: tile(float32, rowStatsL)
      mIj.max(qk, axis = 1, qkMap)
      mIj = max(mI, mIj)                         # merge into the running max
      var alpha: tile(float32, rowStatsL)        # rescale factor (mt × 1)
      alpha = mI - mIj
      alpha.exp2()
      mI = mIj
      qk -.= mIj
      qk.exp2()
      lI *= alpha                                # attention-weight-sum rescale
      var lIj: tile(float32, rowStatsL)          # (mt × 1) attention-weight sum
      lIj.sum(qk, axis = 1, qkMap)
      lI += lIj

      let p = qk.downcast(q.T)                   # P.to(q.T), layout preserved
      acc *.= alpha                              # (mt × D) *.= (mt × 1)
      var vReg: tile(v_pool.T, pvMma.getMmaLayout(opB))  # (D, nt), B operand of MMA 2
      vReg.load(vV, vMap, thrID, (0, 0))
      pvMma.atom.gemm_tile(acc, p, vReg)         # O += P·V   (MMA 2)

    acc /.= lI                                   # (mt × D) /.= (mt × 1)
    segO.store(acc.downcast(o.T), oMap, thrID, (qBlock, 0))

proc attnCombo(o, q, kPool, vPool, qOffsets: distinct TensorView,
               kvOffsets, kvIndices, kvSeqlens: ptr UncheckedArray[int32],
               scale: float32, layer, H, Nkv: int,
               PageSize, D: static int,
               windowed: static bool, windowSize: int) =
  ## Passes the QK mma schedule as the kernel's trailing static.
  ##
  ## Contract: `windowed` and `windowSize` pass explicitly.
  ## The kernel's `mmaQK` has no default. An argument list relying
  ## on the two defaults above them leaves the schedule unbound.
  jagged_attn_fwd(o, q, kPool, vPool, qOffsets, kvOffsets, kvIndices, kvSeqlens,
                  scale, layer, H, Nkv, PageSize, D,
                  windowed, windowSize,
                  tiledMma(float32, q.T, kPool.T, attnTile, attnTile, D))

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  JaggedKvMeta* = object
    ## Host face of the jagged KV metadata, Nim ints, staged to int32 device
    ## buffers per eval, one int32 buffer per array. The page storage belongs
    ## to the PagePool. The metadata walks the kernel to each sequence's
    ## pages inside it.
    ##
    ##   indices  len = capacity                offsets  len = seqlens.len + 1
    ##   [ 0 ][ 2 ][ 0 ][ 1 ][ 3 ]              [ 0 ][ 2 ][ 5 ]
    ##
    ##   seqlens  len = the batch's sequence count
    ##   [ 8 ][ 9 ]
    ##
    ## - the page ids of sequence s are indices[offsets[s] ..< offsets[s + 1]]
    ## - page 0 above serves both sequences, the page ids interleave, each
    ##   new page goes to whichever sequence appends next
    ## - offsets[^1] marks where the newest sequence ends
    ## - seqlens[s] is sequence s's token count
    offsets*: seq[int]
    indices*: seq[int]
    seqlens*: seq[int]

  JaggedAttnNode* = ref object
    ## Jagged attention over the page pool, in graph.
    ## The node's impl is the output o of shape (q_total, H, D) in bfloat16,
    ## a fresh offset-0 cell materialized at enqueue.
    r: AnyTensorImpl
    q: TensorNode                 # (q_total, H, D) post-rope queries
    poolK, poolV: TensorNode      # (num_pages, num_layers, PAGE_SIZE, Nkv, D) slabs
    qOffsets: seq[int]            # per segment q start, seqlens.len + 1 entries
    kv: JaggedKvMeta
    scale: float32
    layer, H, Nkv: int

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*(n: JaggedAttnNode): AnyTensorImpl =
  n.r

func opName*(n: JaggedAttnNode): string =
  "attention"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

func fitsInt32(x: int): bool =
  x >= int32.low.int and x <= int32.high.int

func qOffsetsValid(qOffsets: openArray[int], batchSize, qTotal: int): bool =
  ## True when the q offsets span [0, qTotal) monotonically inside the int32 staging domain
  if qOffsets.len != batchSize + 1 or qOffsets[0] != 0 or
      qOffsets[batchSize] != qTotal:
    return false
  for s in 0 ..< batchSize:
    if qOffsets[s + 1] < qOffsets[s] or not fitsInt32(qOffsets[s + 1]):
      return false
  true

func kvTablesValid(kv: JaggedKvMeta, numPages: int): bool =
  ## True when every segment's KV metadata is consistent:
  ## monotone offsets, positive seqlens, page ids inside the pool,
  ## every staged value inside the int32 domain
  if kv.offsets.len != kv.seqlens.len + 1:
    return false
  if kv.offsets[0] != 0:
    return false
  for s in 0 ..< kv.seqlens.len:
    if kv.seqlens[s] <= 0 or kv.offsets[s + 1] < kv.offsets[s] or
        not fitsInt32(kv.seqlens[s]) or not fitsInt32(kv.offsets[s + 1]):
      return false
    for k in kv.offsets[s] ..< kv.offsets[s + 1]:
      if kv.indices[k] < 0 or kv.indices[k] >= numPages or
          not fitsInt32(kv.indices[k]):
        return false
  true

proc tn_jagged_attention*(engine: AnyEngine, q, poolK, poolV: TensorNode,
                          qOffsets: openArray[int], kv: JaggedKvMeta,
                          scale: float32, layer, H, Nkv: int,
                          windowSize: int = 0): TensorNode {.contracts.} =
  ## Jagged attention over the page pool, in graph. One eval orders
  ## pool writes first, this attention second. The output has shape
  ## (q_total, H, D) in bfloat16 and is fully overwritten.
  ##
  ## Args:
  ##
  ## - q: (q_total, H, D) post-rope queries in bfloat16
  ## - poolK, poolV: (num_pages, num_layers, page_size, Nkv, D) pool slabs
  ##   in bfloat16, the layer selection happens in-kernel through `layer`
  ## - qOffsets: per segment q start, seqlens.len + 1 entries, cumulative,
  ##   qOffsets[^1] == q_total
  ## - kv: the jagged KV metadata, the JaggedKvMeta contract above
  ##
  ## - scale: the softmax scale, 1/sqrt(D) for the Qwen3 geometry
  ## - layer: the pool layer this attention reads, indexed in-kernel
  ## - H, Nkv: query and kv head counts, H must be a multiple of Nkv
  ##
  ## windowSize must be <= 0: this op runs the non-windowed kernel,
  ## Qwen3 has no sliding window.
  ## A windowed dispatch needs a separate op instantiation.
  ##
  ## The op instantiates the Qwen3 statics (page_size 256, head dim 128)
  ## and refuses any other geometry.
  let
    iq = q.getImpl()
    ik = poolK.getImpl()
    iv = poolV.getImpl()
  let
    sq = iq.shape()
    sk = ik.shape()
    sv = iv.shape()
  {.precondition: iq.dtype() == kBf16.}
  {.precondition: ik.dtype() == kBf16.}
  {.precondition: iv.dtype() == kBf16.}
  {.precondition: sq.len == 3.}
  {.precondition: sk.len == 5 and sv.len == 5.}
  {.precondition: sk == sv.}
  {.precondition: sk[2] == PageSize.}
  {.precondition: sq[2] == HeadDim.}
  {.precondition: sq[1] == H and sk[3] == Nkv.}
  {.precondition: H mod Nkv == 0.}
  {.precondition: layer >= 0 and layer < sk[1].}
  {.precondition: windowSize <= 0.}
  {.precondition: kv.seqlens.len > 0.}
  {.precondition: qOffsetsValid(qOffsets, kv.seqlens.len, sq[0]).}
  {.precondition: kvTablesValid(kv, sk[0]).}
  JaggedAttnNode(
    r: engine.newViewNode(kBf16, sq).getImpl(),
    q: q, poolK: poolK, poolV: poolV,
    qOffsets: @qOffsets, kv: kv,
    scale: scale, layer: layer, H: H, Nkv: Nkv).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueJaggedAttnRows[E: HwEngine, Sh: tuple, Mem, MemMeta](
    engine: E, batch: BatchCommands[E], n: JaggedAttnNode,
    ro, rq: TensorImpl[Mem, Layout[Sh, Sh]],
    rk, rv: TensorImpl[Mem, Layout[(int, int, int, int, int),
                                  (int, int, int, int, int)]],
    qOffsetsB: TensorBridge[MemMeta, (int,), (int,)],
    kvOffsets, kvIndices, kvSeqlens: sink MemMeta) =
  ## Enqueues one attnCombo over the node's impls at the Qwen3 statics.
  ##
  ## - the walked body never instantiates at a rank the op does not serve:
  ##   the handler calls it only for rank-3 outputs over rank-5 slabs
  ## - ro is the finalized output buffer. The node's dataless shell stays
  ##   out of the launch site
  when tupleLen(Sh) == 3:
    # one move from each sink parameter into a local. The enqueue macro
    # binds the launch closure's handles from there
    let
      kvOffsetsB = kvOffsets
      kvIndicesB = kvIndices
      kvSeqlensB = kvSeqlens
    let
      ob = initTensorBridge(ro.data[], ro.offset, ro.layout)
      qb = initTensorBridge(rq.data[], rq.offset, rq.layout)
      # pool slabs bridge whole: the kernel indexes the layer-major
      # slab (num_pages, num_layers, page_size, Nkv, D) itself, taking
      # `layer` as a runtime scalar
      kb = initTensorBridge(rk.data[], rk.offset, rk.layout)
      vb = initTensorBridge(rv.data[], rv.offset, rv.layout)
    # launch dims: (q token blocks, H, num_seqs), attnTile lanes
    var maxQLen = 0
    for s in 0 ..< n.kv.seqlens.len:
      maxQLen = max(maxQLen, n.qOffsets[s + 1] - n.qOffsets[s])
    let gx = (maxQLen + attnTile - 1) div attnTile
    batch.enqueue<<((gx, n.H, n.kv.seqlens.len), (attnTile, 1))>> attnCombo(
      ob, qb, kb, vb, qOffsetsB, kvOffsetsB, kvIndicesB, kvSeqlensB,
      float32(n.scale), n.layer, n.H, n.Nkv,
      PageSize, HeadDim, false, low(int32))
  else:
    raise ValueError.newException(
      "eval: the jagged attention op walks rank-3 outputs over rank-5 slabs, " &
      "walked at output rank " & $tupleLen(Sh))

proc enqueueJaggedAttn[E: HwEngine, Sh: tuple, Mem](
    engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes): bool =
  let asA = n.to(JaggedAttnNode)
  if asA.isNil:
    return false
  # Pool children walk first: a pool-write node enqueues its launch here,
  # ordering the write before this attention inside the one batch.
  # Materialized slabs walk as no-ops.
  engine.evalNode(batch, asA.poolK, visited)
  engine.evalNode(batch, asA.poolV, visited)
  engine.evalNode(batch, asA.q, visited)
  # finalization: the output buffer allocates here, a fresh offset-0
  # cell the kernel fully overwrites
  var sh: Sh
  fillShape(sh, asA.r.shape())
  let rr = engine.makeTensorImpl(asA.r.dtype(), asA.r.shape())
    .to(TensorImpl[Mem, Layout[Sh, Sh]])
  let rq = asA.q.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
  type ShP = (int, int, int, int, int)
  var shP: ShP
  fillShape(shP, asA.poolK.getImpl().shape())
  let
    rk = asA.poolK.getImpl().to(TensorImpl[Mem, Layout[ShP, ShP]])
    rv = asA.poolV.getImpl().to(TensorImpl[Mem, Layout[ShP, ShP]])
  # Metadata stages per eval: host Nim ints narrow to int32.
  # A face contract bounds the int32 domain.
  # Kernel binding: the typed device buffers bind the kernel's pointer formals.
  type MemMeta = typeof(engine.allocDevice(int32, 0))
  var qOffHost = newSeq[int32](asA.qOffsets.len)
  for i in 0 ..< qOffHost.len:
    qOffHost[i] = int32(asA.qOffsets[i])
  var kvOffHost = newSeq[int32](asA.kv.offsets.len)
  var kvIdxHost = newSeq[int32](asA.kv.indices.len)
  var kvSeqHost = newSeq[int32](asA.kv.seqlens.len)
  for i in 0 ..< kvOffHost.len:
    kvOffHost[i] = int32(asA.kv.offsets[i])
  for i in 0 ..< kvIdxHost.len:
    kvIdxHost[i] = int32(asA.kv.indices[i])
  for i in 0 ..< kvSeqHost.len:
    kvSeqHost[i] = int32(asA.kv.seqlens[i])
  let
    qOffDev = engine.allocDevice(int32, qOffHost.len)
    kvOffDev = engine.allocDevice(int32, kvOffHost.len)
    kvIdxDev = engine.allocDevice(int32, kvIdxHost.len)
    kvSeqDev = engine.allocDevice(int32, kvSeqHost.len)
  engine.copyToDeviceFromHost(qOffDev, qOffHost)
  engine.copyToDeviceFromHost(kvOffDev, kvOffHost)
  engine.copyToDeviceFromHost(kvIdxDev, kvIdxHost)
  engine.copyToDeviceFromHost(kvSeqDev, kvSeqHost)
  let qOffsetsB = initTensorBridge(qOffDev, 0,
    make_layout((int(qOffHost.len),), (int(1),)))
  # KV metadata DeviceMems bind the kernel's pointer formals directly
  # (D2-8): a DeviceMem-typed struct formal misbinds in the generic
  # handler chain. The per-pointer binding is the mechanism every proven
  # launch on this base uses
  enqueueJaggedAttnRows[E, Sh, Mem, MemMeta](
    engine, batch, asA, rr, rq, rk, rv, qOffsetsB, kvOffDev, kvIdxDev, kvSeqDev)
  n = rr.to(TensorNode)
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("attention", enqueueJaggedAttn)
