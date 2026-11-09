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
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk at import time
## - the Tensor face op over the erased engine
##
## The kernel is positron's `jagged_attn_fwd`, wrapped in the combo proc
## the bf16 smoke proved: the QK mma schedule is the kernel's trailing
## static and must be constructed in the combo body, with `windowed`/
## `windowSize` passed explicitly before it.

from std/math import sqrt, exp
import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/tensorcery/src/device_view
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/tensorcery/src/tensor_datatypes
import workspace/positron/src/kernels/k_jagged_attention
import ../ops_1i1o/rmsnorm

# Batch enqueue walks kernel bodies at compile time and resolves the tile
# algebra's thread->value map templates in the importing module's scope.
export tile_algebra

# Registration order matters: the launch binding of this op's f32 tables
# binds reliably only once the rmsnorm op's walked chain
# (k_rmsnorm -> ceramic rmsnorm) instantiated first in the same binary.
# Importing the rmsnorm op module here registers its handler first,
# whatever order the client imports the op modules.

const attnTile = 32
  ## QK mma schedule tile. page_size 256 is a multiple of the 32-token KV
  ## tile so each KV tile lies inside one page.

type JaggedKvMeta* = object
  ## Host face of the jagged KV metadata, Nim ints, staged to int32 device
  ## buffers per eval. The arrays follow the JaggedKv descriptor contract:
  ##
  ## - the page ids of sequence s are kvIndices[kvOffsets[s] ..< kvOffsets[s + 1]]
  ## - kvOffsets has batchSize + 1 entries and entry batchSize marks where
  ##   the newest sequence ends
  ## - kvSeqlens[s] is sequence s's token count
  offsets*: seq[int]
  indices*: seq[int]
  seqlens*: seq[int]
  batchSize*: int

type MetaCell[Buf] = ref object of RootObj
  ## Owns one staged metadata buffer. RootRef-carried on the node.
  ## A cursor borrow from the enqueue outlives the batch submit.
  buf*: Buf

type JaggedAttnNode* = ref object
  ## Jagged attention over the page pool, in graph. The node's impl is
  ## the output o of shape (q_total, H, D) in bfloat16.
  r: AnyTensorImpl
  q: TensorNode                 # (q_total, H, D) post-rope queries
  poolK, poolV: TensorNode      # (num_pages, num_layers, page_size, Nkv, D) slabs
  qOffsets: seq[int]            # per segment q start, batchSize + 1 entries
  qOffsetsCell: RootRef         # MetaCell box, staged per eval, node-owned.
                                # Launch closures run after handler return,
                                # so a cursor borrow from the enqueue must
                                # outlive the batch submit
  kv: JaggedKvMeta
  scale: float32
  layer, H, Nkv: int
  windowSize: int               # must be <= 0, the op runs non-windowed
  enqueued: bool
    ## Launch-once marker under the eval walk's fan-out re-walks:
    ## - a downstream op that fans out over a shared input re-walks
    ##   the whole upstream chain (the FFN's two projections share one input)
    ## - the repeat visit must not re-run the handler: re-staging
    ##   the metadata cell would enqueue the launch twice
    ## Precondition: an op node is constructed per forward and enqueued in
    ## exactly one eval, so the flag never crosses an eval boundary.

proc eval*(n: JaggedAttnNode) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getImpl*(n: JaggedAttnNode): AnyTensorImpl =
  n.r

# ═══════════════════════════════════════════════════════════════════════
#  Eval - enqueue the node's kernel into the walk's batch
# ═══════════════════════════════════════════════════════════════════════

type JaggedKvBridge[DM] = object
  ## Launch argument for JaggedKv: one buffer binding per metadata array,
  ## batchSize packs as a scalar. The fields keep the JaggedKv descriptor's
  ## order and the launch struct maps them positionally.
  offsets*, indices*, seqlens*: DM
  batchSize*: int

type BridgeTensor[DevMem, Sh, St] = object
  ## Launch argument for a `distinct TensorView` kernel formal:
  ##
  ## - data: the whole device buffer, no offset
  ## - layout: the view layout (view origin at the buffer base)
  ##
  ## The launch explodes the DevMem leaf to a buffer binding.
  data* {.cursor.}: DevMem
  layout*: Layout[Sh, St]

proc enqueueJaggedAttnNode*[E: HwEngine, Sh: tuple](engine: E, batch: Batch[E], n: TensorNode): bool =
  ## Enqueue handler of the jagged attention op, registered below.
  ## The erased node downcasts to the attention node here.
  ## A node of another op returns false, leaving the dispatch to that op's handler.
  let asA = n.to(JaggedAttnNode)
  if asA.isNil:
    return false
  if asA.enqueued:
    return true
  proc attnCombo(o, q, kPool, vPool, qOffsets: distinct TensorView,
                 kv: JaggedKv, scale: float32, layer, H, Nkv: int,
                 PageSize, D: static int,
                 windowed: static bool, windowSize: int) =
    ## Passes the QK mma schedule as the kernel's trailing static.
    ##
    ## Contract: `windowed` and `windowSize` pass explicitly.
    ## The kernel's `mmaQK` has no default. An argument list relying
    ## on the two defaults above them leaves the schedule unbound.
    jagged_attn_fwd(o, q, kPool, vPool, qOffsets, kv,
                    scale, layer, H, Nkv, PageSize, D,
                    windowed, windowSize,
                    tiledMma(float32, q.T, kPool.T, attnTile, attnTile, D))

  proc enqueueJaggedAttn[E: HwEngine, Mem, MemMeta](
      engine: E, batch: Batch[E], n: JaggedAttnNode) =
    ## Enqueues one attnCombo over the node's impls at the Qwen3 statics.
    if n.enqueued:
      return
    ##
    ## Literal shape tuples at the call site are the erasedUpload pattern.
    ## Sh-parametric type expressions bind unreliably at launch time.
    ## Literal types always bind.
    ##
    ## The pools bridge whole. The kernel indexes the layer-major slab
    ## (num_pages, num_layers, page_size, Nkv, D) itself, taking
    ## `layer` as a runtime scalar.
    let
      io = n.r.to(TensorImpl[Mem, Layout[(int, int, int), (int, int, int)]])
      iq = n.q.getImpl().to(TensorImpl[Mem, Layout[(int, int, int), (int, int, int)]])
      ik = n.poolK.getImpl().to(TensorImpl[Mem, Layout[(int, int, int, int, int), (int, int, int, int, int)]])
      iv = n.poolV.getImpl().to(TensorImpl[Mem, Layout[(int, int, int, int, int), (int, int, int, int, int)]])
    let
      ob = BridgeTensor[Mem, (int, int, int), (int, int, int)](data: io.data[], layout: io.layout)
      qb = BridgeTensor[Mem, (int, int, int), (int, int, int)](data: iq.data[], layout: iq.layout)
      kb = BridgeTensor[Mem, (int, int, int, int, int), (int, int, int, int, int)](data: ik.data[], layout: ik.layout)
      vb = BridgeTensor[Mem, (int, int, int, int, int), (int, int, int, int, int)](data: iv.data[], layout: iv.layout)
    # q_offsets staged per eval into the node's cell: host Nim ints,
    # int32 on device, rank 1. The bridge cursor-borrows the cell buffer,
    # and the node keeps it alive through the batch submit.
    let qOffCell = cast[MetaCell[MemMeta]](n.qOffsetsCell)
    let qOffB = BridgeTensor[MemMeta, (int,), (int,)](
      data: qOffCell.buf,
      layout: make_layout((int(n.kv.batchSize + 1),), (int(1),)))
    # KV metadata staged per eval, one buffer binding per array
    let
      offDev = engine.allocDevice(int32, n.kv.offsets.len)
      idxDev = engine.allocDevice(int32, n.kv.indices.len)
      seqDev = engine.allocDevice(int32, n.kv.seqlens.len)
    var
      offHost = newSeq[int32](n.kv.offsets.len)
      idxHost = newSeq[int32](n.kv.indices.len)
      seqHost = newSeq[int32](n.kv.seqlens.len)
    for i in 0 ..< n.kv.offsets.len:
      offHost[i] = int32(n.kv.offsets[i])
    for i in 0 ..< n.kv.indices.len:
      idxHost[i] = int32(n.kv.indices[i])
    for i in 0 ..< n.kv.seqlens.len:
      seqHost[i] = int32(n.kv.seqlens[i])
    engine.copyToDeviceFromHost(offDev, offHost)
    engine.copyToDeviceFromHost(idxDev, idxHost)
    engine.copyToDeviceFromHost(seqDev, seqHost)
    let kvB = JaggedKvBridge[typeof(offDev)](
      offsets: offDev, indices: idxDev, seqlens: seqDev,
      batchSize: n.kv.batchSize)
    # launch dims: (q token blocks, H, num_seqs), 32 lanes
    var maxQLen = 0
    for s in 0 ..< n.kv.batchSize:
      maxQLen = max(maxQLen, n.qOffsets[s + 1] - n.qOffsets[s])
    let gx = (maxQLen + attnTile - 1) div attnTile
    let windowSize = n.windowSize
    # One call site only: two walked enqueues of one kernel fn corrupt the launch.
    # Batcher compilation keeps one artifact per kernel fn, and the last
    # enqueue wins (see the multi-enqueue reproducer).
    # Qwen3 runs non-windowed (no sliding window).
    # A windowed dispatch needs a separate op instantiation.
    n.enqueued = true
    batch.enqueue<<((gx, n.H, n.kv.batchSize), (attnTile, 1))>> attnCombo(ob, qb, kb, vb, qOffB, kvB, float32(n.scale), n.layer, n.H, n.Nkv, 256, 128, false, low(int32))


  # Pool children walk first: a pool-write node enqueues its launch here,
  # ordering the write before this attention inside the one batch.
  # Materialized slabs walk as no-ops.
  evalNode[E](engine, batch, asA.poolK)
  evalNode[E](engine, batch, asA.poolV)
  evalNode[E](engine, batch, asA.q)
  # q_offsets staged per eval in the handler, outside the walked body,
  # per the embed gather ids pattern
  var qOffHost = newSeq[int32](asA.kv.batchSize + 1)
  for i in 0 ..< qOffHost.len:
    qOffHost[i] = int32(asA.qOffsets[i])
  type MemMeta = typeof(engine.allocDevice(int32, 0))
  var cell = MetaCell[MemMeta]()
  cell.buf = engine.allocDevice(int32, qOffHost.len)
  engine.copyToDeviceFromHost(cell.buf, qOffHost)
  asA.qOffsetsCell = RootRef(cell)
  # Qwen3 statics: page_size 256, D 128, bf16 mma tile 32.
  # Kernel schedules are compile-time.
  # This op instantiates that geometry and raises on anything else.
  let poolShape = asA.poolK.getImpl().shape()
  let qShape = asA.q.getImpl().shape()
  if poolShape.len != 5 or poolShape[2] != 256 or qShape[2] != 128:
    raise ValueError.newException(
      "eval: the jagged attention op instantiates the Qwen3 geometry " &
      "(page_size 256, D 128) over rank-5 slabs, walked at (" &
      $poolShape & ", " & $qShape[2] & ")")
  enqueueJaggedAttn[E, typeof(engine.allocDevice(bfloat16, 0)),
                    typeof(engine.allocDevice(int32, 0))](engine, batch, asA)
  return true

registerEnqueueNode(enqueueJaggedAttnNode)

# ═══════════════════════════════════════════════════════════════════════
#  Tensor face
# ═══════════════════════════════════════════════════════════════════════

proc tn_jagged_attention*(engine: AnyEngine, q, poolK, poolV: TensorNode,
                          qOffsets: openArray[int], kv: JaggedKvMeta,
                          scale: float32, layer, H, Nkv: int,
                          windowSize: int = 0): TensorNode =
  ## Jagged attention over the page pool, in graph. One eval orders
  ## the pool write first and this attention second. The output has
  ## shape (q_total, H, D), is bfloat16 and is fully overwritten.
  ##
  ## Args:
  ##
  ## - q: (q_total, H, D) post-rope queries in bfloat16
  ## - poolK, poolV: (num_pages, num_layers, page_size, Nkv, D) pool slabs
  ##   in bfloat16, the layer selection happens in-kernel through `layer`
  ## - qOffsets: per segment q start, batchSize + 1 entries, cumulative,
  ##   qOffsets[batchSize] == q_total
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
  ## Raises on a rank/dtype/geometry mismatch.
  ## Raises on qOffsets or KV metadata inconsistencies.
  ## Raises on windowSize > 0 and outside the Qwen3 statics (page_size 256, head dim 128).
  let
    iq = q.getImpl()
    ik = poolK.getImpl()
    iv = poolV.getImpl()
  for (node, name) in [(iq, "q"), (ik, "poolK"), (iv, "poolV")]:
    if node.dtype() != kBf16:
      raise ValueError.newException(
        "tn_jagged_attention: " & name & " dtype " & $node.dtype() &
        " unsupported, the attention runs bf16 in and out")
  let
    sq = iq.shape()
    sk = ik.shape()
    sv = iv.shape()
  if sq.len != 3:
    raise ValueError.newException(
      "tn_jagged_attention: q must be rank 3 (q_total, H, D), got rank " & $sq.len)
  if sk.len != 5 or sv.len != 5:
    raise ValueError.newException(
      "tn_jagged_attention: the pool slabs must be rank 5 " &
      "(num_pages, num_layers, page_size, Nkv, D), got rank " &
      $sk.len & " and " & $sv.len)
  if sk != sv:
    raise ValueError.newException(
      "tn_jagged_attention: pool geometry mismatch, got " & $sk & " and " & $sv)
  if sq[2] != sk[4]:
    raise ValueError.newException(
      "tn_jagged_attention: q head dim " & $sq[2] & " != pool head dim " & $sk[4])
  if sk[2] mod attnTile != 0:
    raise ValueError.newException(
      "tn_jagged_attention: page_size " & $sk[2] &
      " must be a multiple of the " & $attnTile & "-token KV tile")
  if sq[1] != H or sk[3] != Nkv:
    raise ValueError.newException(
      "tn_jagged_attention: head counts (" & $sq[1] & ", " & $sk[3] &
      ") mismatch the face args (" & $H & ", " & $Nkv & ")")
  if layer < 0 or layer >= sk[1]:
    raise ValueError.newException(
      "tn_jagged_attention: layer " & $layer & " outside the slabs' " &
      $sk[1] & " layers")
  if H mod Nkv != 0:
    raise ValueError.newException(
      "tn_jagged_attention: H " & $H & " must be a multiple of Nkv " & $Nkv)
  if windowSize > 0:
    raise ValueError.newException(
      "tn_jagged_attention: the windowed dispatch needs its own op " &
      "instantiation, this op runs non-windowed (Qwen3 has no sliding window)")
  if kv.batchSize <= 0:
    raise ValueError.newException(
      "tn_jagged_attention: batchSize " & $kv.batchSize & " must be positive")
  if qOffsets.len != kv.batchSize + 1:
    raise ValueError.newException(
      "tn_jagged_attention: qOffsets carry " & $qOffsets.len &
      " entries, batchSize + 1 = " & $(kv.batchSize + 1))
  if qOffsets[0] != 0 or qOffsets[kv.batchSize] != sq[0]:
    raise ValueError.newException(
      "tn_jagged_attention: qOffsets must span [0, " & $sq[0] & "), got [" &
      $qOffsets[0] & ", " & $qOffsets[kv.batchSize] & "]")
  for s in 0 ..< kv.batchSize:
    if qOffsets[s + 1] < qOffsets[s]:
      raise ValueError.newException(
        "tn_jagged_attention: qOffsets not monotone at segment " & $s)
    if kv.seqlens[s] <= 0:
      raise ValueError.newException(
        "tn_jagged_attention: seqlens[" & $s & "] = " & $kv.seqlens[s] &
        " must be positive")
    if kv.offsets[s + 1] < kv.offsets[s]:
      raise ValueError.newException(
        "tn_jagged_attention: kv offsets not monotone at segment " & $s)
    for k in kv.offsets[s] ..< kv.offsets[s + 1]:
      if kv.indices[k] < 0 or kv.indices[k] >= sk[0]:
        raise ValueError.newException(
          "tn_jagged_attention: page id " & $kv.indices[k] &
          " outside the pool of " & $sk[0] & " pages")
  JaggedAttnNode(
    r: engine.newTensorNode(kBf16, sq).getImpl(),
    q: q, poolK: poolK, poolV: poolV,
    qOffsets: @qOffsets, kv: kv,
    scale: scale, layer: layer, H: H, Nkv: Nkv,
    windowSize: windowSize).to(TensorNode)
