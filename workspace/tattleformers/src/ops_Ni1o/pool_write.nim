# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
## Page pool write op: kernel, graph node and Tensor face op colocated.
##
## Same file shape as the rope op:
##
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk at import time
## - the Tensor face op over the erased engine
##
## The kernel is positron's `k_pool_write`: the pool slot layout is
## transformer-specialized, so the kernel lives in positron.

import std/math
import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/tensorcery/src/tensor_datatypes
import workspace/positron/src/kernels/k_pool_write
import ../ops_1i1o/rmsnorm

# Batch enqueue walks kernel bodies at compile time and resolves the tile
# algebra's thread->value map templates in the importing module's scope.
export tile_algebra

# Registration order matters: the launch binding of this op's bridges
# binds reliably only once the rmsnorm op's walked chain
# (k_rmsnorm -> ceramic rmsnorm) instantiated first in the same binary.
# Importing the rmsnorm op module here registers its handler first,
# whatever order the client imports the op modules.

type PoolWriteNode* = ref object
  ## Writes post-rope K/V entries into pool pages, in graph.
  ## The node's impl is the K pool layer view: the output aliases the pool,
  ## downstream nodes read the written slots through it.
  r: AnyTensorImpl
  poolK, poolV: TensorNode # (num_pages, PAGE_SIZE, kv_heads, head_dim) layer views
  kSrc, vSrc: TensorNode   # (1, seq_len, kv_heads, head_dim) post-rope entries
  offset: int              # global token position of src entry 0
  seqLen: int              # entries to write
  cachedTokens: int        # entries already in the trie from copy-on-write forks
  pageSlots: seq[int]      # pool slot of each sequence page, in page order

proc eval*(n: PoolWriteNode) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getImpl*(n: PoolWriteNode): AnyTensorImpl =
  n.r

# ═══════════════════════════════════════════════════════════════════════
#  Eval - enqueue the node's kernel into the walk's batch
# ═══════════════════════════════════════════════════════════════════════

const poolWriteBlk = 256

proc enqueuePoolWriteRows[E: HwEngine, Sh: tuple, Mem, MemMeta](
    engine: E, batch: Batch[E], n: PoolWriteNode, meta: sink MemMeta) =
  ## Enqueues one k_pool_write over the node's write range.
  ## A body-local type alias defeats the launch's generic binding.
  let
    pk = n.poolK.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    pv = n.poolV.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    sk = n.kSrc.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    sv = n.vSrc.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
  let
    pageSize = pk.layout.shape[1]
    kvHeads = pk.layout.shape[2]
    headDim = pk.layout.shape[3]
    seqLen = sk.layout.shape[1]
    dstK = TensorBridge[Mem, Sh, Sh](data: pk.data[], offset: pk.offset, layout: pk.layout)
    dstV = TensorBridge[Mem, Sh, Sh](data: pv.data[], offset: pv.offset, layout: pv.layout)
    # Src entries (1, seq_len, kv_heads, head_dim), batch 1 per forward pass.
    # Kernel side: one (seq_len, kv_heads, head_dim) window at the batch offset.
    srcK = TensorBridge[Mem, (int, int, int), (int, int, int)](
      data: sk.data[], offset: sk.offset,
      layout: make_layout((int(seqLen), int(kvHeads), int(headDim)),
        (int(sk.layout.stride[1]), int(sk.layout.stride[2]), int(sk.layout.stride[3]))))
    srcV = TensorBridge[Mem, (int, int, int), (int, int, int)](
      data: sv.data[], offset: sv.offset,
      layout: make_layout((int(seqLen), int(kvHeads), int(headDim)),
        (int(sv.layout.stride[1]), int(sv.layout.stride[2]), int(sv.layout.stride[3]))))
  # One move from the sink parameter into a local: the enqueue macro binds
  # its launch closure's handle from that local.
  let metaBuf = meta
  let numel = seqLen * kvHeads * headDim
  batch.enqueue<<((numel + poolWriteBlk - 1) div poolWriteBlk, poolWriteBlk)>> k_pool_write(dstK, srcK, dstV, srcV, metaBuf)

proc enqueuePoolWriteNode*[E: HwEngine, Sh: tuple](engine: E, batch: Batch[E], n: TensorNode): bool =
  ## Enqueue handler of the pool write op, registered below. The erased node
  ## downcasts to the pool write node here. A node of another op returns false,
  ## leaving the dispatch to that op's handler.
  let asP = n.to(PoolWriteNode)
  if asP.isNil:
    return false
  evalNode[E](engine, batch, asP.kSrc)
  evalNode[E](engine, batch, asP.vSrc)
  # Pool views and src entries share one bf16 Mem (checked at the face).
  # Rank dispatch: the enqueue template runs at the rank 4 branch only,
  # one literal shape tuple per branch, the rmsnorm handler pattern.
  if asP.poolK.getImpl().shape().len != 4:
    raise ValueError.newException(
      "eval: the pool write op is rank 4, walked at rank " &
      $asP.poolK.getImpl().shape().len)
  type
    Mem = typeof(engine.allocDevice(bfloat16, 0))
  # Write range metadata: Nim ints at the host face, int32 on device,
  # (offset, seqLen, cachedTokens, pageSize, pageSlots[0], pageSlots[1], ...),
  # staged per eval
  let pkShape = asP.poolK.getImpl().shape()
  var metaHost = newSeq[int32](4 + asP.pageSlots.len)
  metaHost[0] = int32(asP.offset)
  metaHost[1] = int32(asP.seqLen)
  metaHost[2] = int32(asP.cachedTokens)
  metaHost[3] = int32(pkShape[1])
  for i in 0 ..< asP.pageSlots.len:
    metaHost[4 + i] = int32(asP.pageSlots[i])
  let metaDev = engine.allocDevice(int32, metaHost.len)
  engine.copyToDeviceFromHost(metaDev, metaHost)
  enqueuePoolWriteRows[E, (int, int, int, int), Mem, typeof(metaDev)](
    engine, batch, asP, metaDev)
  return true

registerEnqueueNode(enqueuePoolWriteNode)

# ═══════════════════════════════════════════════════════════════════════
#  Tensor face
# ═══════════════════════════════════════════════════════════════════════

proc tn_pool_write*(engine: AnyEngine, poolK, poolV, kSrc, vSrc: TensorNode, offset, seqLen, cachedTokens: int, pageSlots: openArray[int]): TensorNode =
  ## Writes post-rope K/V entries into pool pages, in graph: the write enqueues
  ## before any node reading the pool, one eval orders write then attention.
  ## The returned node aliases the K pool view, the V pool view is written
  ## by the same launch.
  ##
  ## Args:
  ## - poolK, poolV: (num_pages, PAGE_SIZE, kv_heads, head_dim) layer views
  ##   of the pool slabs in bfloat16, from PagePool.layerView
  ## - kSrc, vSrc: (1, seq_len, kv_heads, head_dim) post-rope entries,
  ##   bfloat16, batch index 0 (batch 1 per forward pass)
  ## - offset: global token position of src entry 0
  ## - pageSlots: the pool slot of each sequence page, in page order,
  ##   pageSlots[globalPos div PAGE_SIZE] receives globalPos's entries,
  ##   the slots sit anywhere in the pool (trie-cached pages included)
  ##
  ## Ranges:
  ## - seqLen: entries to write, seqLen <= kSrc's token count
  ## - cachedTokens: entries already in the trie from copy-on-write forks,
  ##   positions before max(0, cachedTokens - offset) stay untouched
  ##
  ## Raises on a rank/dtype/geometry mismatch and when the write range
  ## exceeds the pool capacity.
  let
    ipk = poolK.getImpl()
    ipv = poolV.getImpl()
    isk = kSrc.getImpl()
    isv = vSrc.getImpl()
  for (node, name) in [(ipk, "poolK"), (ipv, "poolV"), (isk, "kSrc"), (isv, "vSrc")]:
    if node.dtype() != kBf16:
      raise ValueError.newException(
        "tn_pool_write: " & name & " dtype " & $node.dtype() &
        " unsupported, the pool write runs bf16 in and out")
  let
    spk = ipk.shape()
    spv = ipv.shape()
    ssk = isk.shape()
    ssv = isv.shape()
  if spk.len != 4 or spv.len != 4:
    raise ValueError.newException(
      "tn_pool_write: the pool views must be rank 4 " &
      "(num_pages, PAGE_SIZE, kv_heads, head_dim), got rank " &
      $spk.len & " and " & $spv.len)
  if ssk.len != 4 or ssv.len != 4:
    raise ValueError.newException(
      "tn_pool_write: the src rows must be rank 4 (1, seq_len, kv_heads, head_dim), " &
      "got rank " & $ssk.len & " and " & $ssv.len)
  if spk != spv:
    raise ValueError.newException(
      "tn_pool_write: pool geometry mismatch, got " & $spk & " and " & $spv)
  if ssk != ssv:
    raise ValueError.newException(
      "tn_pool_write: src geometry mismatch, got " & $ssk & " and " & $ssv)
  if ssk[0] != 1:
    raise ValueError.newException(
      "tn_pool_write: the src rows carry batch " & $ssk[0] & ", batch 1 per forward pass")
  if ssk[2] != spk[2] or ssk[3] != spk[3]:
    raise ValueError.newException(
      "tn_pool_write: layer kv geometry (" & $ssk[2] & ", " & $ssk[3] &
      ") exceeds the pool slot (" & $spk[2] & ", " & $spk[3] & ")")
  if seqLen < 0 or seqLen > ssk[1]:
    raise ValueError.newException(
      "tn_pool_write: seqLen " & $seqLen & " outside the src rows' " & $ssk[1] & " tokens")
  if offset < 0 or cachedTokens < 0:
    raise ValueError.newException(
      "tn_pool_write: negative offset " & $offset & " or cachedTokens " & $cachedTokens)
  let pageSize = spk[1]
  let lastPos = offset + seqLen - 1
  if seqLen > 0 and lastPos div pageSize >= pageSlots.len:
    raise ValueError.newException(
      "tn_pool_write: write range reaches sequence page " & $(lastPos div pageSize) &
      ", the page table lists " & $pageSlots.len & " pages")
  for i in 0 ..< pageSlots.len:
    if pageSlots[i] < 0 or pageSlots[i] >= spk[0]:
      raise ValueError.newException(
        "tn_pool_write: page table slot " & $pageSlots[i] &
        " outside the pool of " & $spk[0] & " pages")
  PoolWriteNode(
    r: ipk,
    poolK: poolK, poolV: poolV, kSrc: kSrc, vSrc: vSrc,
    offset: offset, seqLen: seqLen, cachedTokens: cachedTokens,
    pageSlots: @pageSlots).to(TensorNode)
