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
## - the device kernel
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk by registerOp
## - the Tensor face op over the erased engine
##
## One launch writes the whole range. The kernel decomposes pages:
##
## - sequence page = globalPos div pageSize
## - pool slot = the metadata table entry for that sequence page
##
## Values are copied bit-exact, no arithmetic.
##
## Output binding:
##
## - the node's impl is the pool K slab's own impl, already materialized at offset 0
## - the handler allocates nothing and constructs no new output
## - after the enqueue the handler replaces the walked node with the slab leaf
## - layer and page-geometry offsets live in the kernel's TensorBridge
##   arguments (launch ABI), never in a TensorImpl
##
## Consumers read the written pool through the replaced node.
## No TensorImpl leaves this op with a runtime offset: every impl's offset stays final at its face.
##
## Lifetime:
##
## - the launch closure captures only the TensorBridges and the staged int32 metadata DeviceMem
## - the batch's borrow tracking retains device buffers at enqueue, releasing them on batch destroy, so staged buffers survive past submit
## - the metadata's host values stage into the device buffer before the enqueue, nothing on the Nim side is captured

import std/typetraits
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/crucible/src/builtins/builtins_gpu_types
import workspace/iface
import ../../tensors_internals
import ../dispatch
import workspace/design_by_contract

# Batch enqueue walks the kernel body at compile time and resolves the tile
# algebra's thread-coordinate templates in the importing module's scope.
export tile_algebra

# ############################################################
#
#                   Pool write
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_pool_write*(dstK, srcK, dstV, srcV: distinct TensorView, meta: ptr UncheckedArray[int32]) =
  ## Copies post-rope K/V entries into their pool slots, both pool windows
  ## fully overwritten over the covered positions.
  ##
  ## Expected input:
  ## - srcK, srcV: (seq_len, kv_heads, head_dim) entries of the post-rope K and V
  ## - dstK, dstV: (num_pages, page_size, kv_heads, head_dim) pool layer views
  ## - meta: int32 metadata, entry layout: offset, seqLen, cachedTokens,
  ##   pageSize, then one pool slot per sequence page
  ##
  ## Output:
  ## - dstK, dstV fully written over positions [offset, offset + seq_len)
  ## - source element (t, h, d) writes pool entry (page, withinPage, h, d) at slot meta[4 + globalPos div page_size]
  ##
  ## Positions before max(0, cachedTokens - offset) keep their previous content.
  ## Definitions:
  ##
  ## - sequence page = globalPos div page_size
  ## - withinPage = globalPos mod page_size
  ## - globalPos = offset + t
  ##
  ## Grid-stride contract:
  ##
  ## - tid = thread_position_in_grid spans threadgroups x threads entries
  ##   and strides by exactly that product, so each element is written once
  ## - a stride of threadgroups alone makes every thread with the same
  ##   threadgroup-local index write the same elements concurrently
  let
    offset = meta[0]
    seqLen = meta[1]
    cachedTokens = meta[2]
    pageSize = meta[3]
    headDim = srcK.layout.shape[2]
    kvHeadDim = srcK.layout.shape[1] * headDim
    numel = srcK.layout.shape[0] * kvHeadDim
    tid = thread_position_in_grid.x
    stride = threadgroups_per_grid.x * threads_per_threadgroup.x
  let writeStart = max(0'i32, cachedTokens - offset)

  for i in countup(tid, int32(numel - 1), stride):
    let
      t = i div int32(kvHeadDim)
      rem = i mod int32(kvHeadDim)
      h = rem div int32(headDim)
      d = rem mod int32(headDim)
    if t >= writeStart:
      let
        globalPos = offset + t
        page = meta[4 + globalPos div pageSize]
        withinPage = globalPos mod pageSize
      dstK[(int(page), int(withinPage), int(h), int(d))] = srcK[(int(t), int(h), int(d))]
      dstV[(int(page), int(withinPage), int(h), int(d))] = srcV[(int(t), int(h), int(d))]

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  PoolWriteNode* = ref object
    ## Writes post-rope K/V entries into pool pages, in graph.
    ## The node's impl is the K pool slab: the output aliases the pool.
    r: AnyTensorImpl
    poolK, poolV: TensorNode # (num_pages, num_layers, PAGE_SIZE, kv_heads, head_dim) slabs
    layer: int               # target slab layer
    kSrc, vSrc: TensorNode   # (1, seq_len, kv_heads, head_dim) post-rope entries
    offset: int              # global token position of src entry 0
    seqLen: int              # entries to write
    cachedTokens: int        # entries already cached from copy-on-write forks
    pageSlots: seq[int]      # pool slot of each sequence page, in page order

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*(n: PoolWriteNode): AnyTensorImpl =
  n.r

func opName*(n: PoolWriteNode): string =
  "pool_write"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

func fitsInt32(x: int): bool =
  x >= int32.low.int and x <= int32.high.int

func pageSlotsInPool(pageSlots: openArray[int], numPages: int): bool =
  ## Every slot selects an existing pool page and fits the int32 metadata
  for i in 0 ..< pageSlots.len:
    if pageSlots[i] < 0 or pageSlots[i] >= numPages or not fitsInt32(pageSlots[i]):
      return false
  true

proc tn_pool_write*(engine: AnyEngine, poolK, poolV, kSrc, vSrc: TensorNode,
    layer: int, offset, seqLen, cachedTokens: int, pageSlots: openArray[int]): TensorNode {.contracts.} =
  ## Writes post-rope K/V entries into pool pages, in graph.
  ##
  ## - the write enqueues before any node reading the pool
  ## - one eval orders write then read through the child walk
  ## - the returned node aliases the K pool SLAB, so a consumer that reads
  ##   the written pool takes the returned node as its child
  ## - one launch writes both pools
  ##
  ## Args:
  ## - poolK, poolV: (num_pages, num_layers, PAGE_SIZE, kv_heads, head_dim)
  ##   pool slabs in bfloat16. `layer` selects the slab layer the write targets.
  ## - kSrc, vSrc: (1, seq_len, kv_heads, head_dim) post-rope entries, bfloat16, batch index 0
  ## - offset: global token position of src entry 0
  ## - pageSlots: the pool slot of each sequence page, in page order
  ##
  ## Table semantics:
  ##
  ## - pageSlots[globalPos div PAGE_SIZE] receives globalPos's entries
  ## - a trie-cached page occupies any pool slot
  ##
  ## Ranges:
  ## - seqLen: entries to write, at least one, at most kSrc's token count
  ## - seqLen 0 would launch zero threadgroups and is refused
  ## - cachedTokens: entries already in the cache from copy-on-write forks,
  ##   positions before max(0, cachedTokens - offset) stay untouched
  ## - the int32 metadata domain bounds every staged value: offset, seqLen,
  ##   cachedTokens, pageSize, each page slot, offset + seqLen
  let
    ipk = poolK.getImpl()
    ipv = poolV.getImpl()
    isk = kSrc.getImpl()
    isv = vSrc.getImpl()
  let
    spk = ipk.shape()
    spv = ipv.shape()
    ssk = isk.shape()
    ssv = isv.shape()
  {.precondition: ipk.dtype() == kBf16.}
  {.precondition: ipv.dtype() == kBf16.}
  {.precondition: isk.dtype() == kBf16.}
  {.precondition: isv.dtype() == kBf16.}
  {.precondition: spk.len == 5 and spv.len == 5.}
  {.precondition: ssk.len == 4 and ssv.len == 4.}
  {.precondition: spk == spv.}
  {.precondition: ssk == ssv.}
  {.precondition: ssk[0] == 1.}
  {.precondition: layer >= 0 and layer < spk[1].}
  {.precondition: ssk[2] == spk[3] and ssk[3] == spk[4].}
  {.precondition: seqLen >= 1 and seqLen <= ssk[1].}
  {.precondition: offset >= 0 and cachedTokens >= 0.}
  {.precondition: (offset + seqLen - 1) div spk[2] < pageSlots.len.}
  {.precondition: pageSlotsInPool(pageSlots, spk[0]).}
  {.precondition: fitsInt32(offset) and fitsInt32(seqLen) and fitsInt32(cachedTokens).}
  {.precondition: fitsInt32(spk[2]) and fitsInt32(offset + seqLen).}
  PoolWriteNode(
    r: ipk,
    poolK: poolK, poolV: poolV, layer: layer, kSrc: kSrc, vSrc: vSrc,
    offset: offset, seqLen: seqLen, cachedTokens: cachedTokens,
    pageSlots: @pageSlots).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

const poolWriteBlk = 256

proc enqueuePoolWriteRows[E: HwEngine, ShP: tuple, ShS: tuple, Mem, MemMeta](
    engine: E, batch: BatchCommands[E], n: PoolWriteNode,
    pk, pv: TensorImpl[Mem, Layout[ShP, ShP]],
    sk, sv: TensorImpl[Mem, Layout[ShS, ShS]],
    meta: sink MemMeta) =
  ## Enqueues one k_pool_write over the node's impls.
  ##
  ## Args:
  ## - pk, pv, sk, sv: the materialized impls, the pool impl binds the launch directly
  ## - meta: the staged int32 metadata DeviceMem
  ##
  ## - no output allocation happens here
  ## - a body-local type alias defeats the launch's generic binding
  ## - the walked body never instantiates at a rank pair the op does not serve, the handler calls it only for rank-5 slabs over rank-4 src
  when tupleLen(ShP) == 5 and tupleLen(ShS) == 4:
    let
      slab = pk.layout
      # Dst = the layer's (num_pages, PAGE_SIZE, kv_heads, head_dim) window:
      # layer axis drops, offset is the layer base
      dstLayout = make_layout(
        (int(slab.shape[0]), int(slab.shape[2]), int(slab.shape[3]), int(slab.shape[4])),
        (int(slab.stride[0]), int(slab.stride[2]), int(slab.stride[3]), int(slab.stride[4])))
      dstOffset = pk.offset + n.layer * int(slab.stride[1])
      pageSize = int(slab.shape[2])
      kvHeads = int(slab.shape[3])
      headDim = int(slab.shape[4])
      seqLen = int(sk.layout.shape[1])
      dstK = TensorBridge[Mem, (int, int, int, int), (int, int, int, int)](
        data: pk.data[], offset: dstOffset, layout: dstLayout)
      dstV = TensorBridge[Mem, (int, int, int, int), (int, int, int, int)](
        data: pv.data[], offset: dstOffset, layout: dstLayout)
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
    # one move from the sink parameter into a local. The enqueue macro binds the launch closure's handle from there.
    let metaBuf = meta
    let numel = seqLen * kvHeads * headDim
    batch.enqueue<<((numel + poolWriteBlk - 1) div poolWriteBlk, poolWriteBlk)>> k_pool_write(dstK, srcK, dstV, srcV, metaBuf)
  else:
    raise ValueError.newException(
      "eval: the pool write op walks rank-5 slabs over rank-4 src rows, walked at ranks " &
      $tupleLen(ShP) & " and " & $tupleLen(ShS))

proc enqueuePoolWrite[E: HwEngine, Sh: tuple, Mem](
    engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes): bool =
  let asP = n.to(PoolWriteNode)
  if asP.isNil:
    return false
  engine.evalNode(batch, asP.kSrc, visited)
  engine.evalNode(batch, asP.vSrc, visited)
  # Output cell: the pool K slab's own impl, already materialized.
  # No allocation, no fresh output ctor. The pool slabs are leaves.
  var sh: Sh
  fillShape(sh, asP.r.shape())
  let
    pk = asP.poolK.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    pv = asP.poolV.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
  # src entries are rank 4 over the same Mem (dtype equality checked at the face)
  type ShS = (int, int, int, int)
  var shS: ShS
  fillShape(shS, asP.kSrc.getImpl().shape())
  let
    sk = asP.kSrc.getImpl().to(TensorImpl[Mem, Layout[ShS, ShS]])
    sv = asP.vSrc.getImpl().to(TensorImpl[Mem, Layout[ShS, ShS]])
  # Staging of write-range metadata per eval: host Nim ints narrow to int32, a face contract bounds the int32 domain.
  # Kernel binding: the typed device buffer binds the kernel's pointer formal.
  var metaHost = newSeq[int32](4 + asP.pageSlots.len)
  metaHost[0] = int32(asP.offset)
  metaHost[1] = int32(asP.seqLen)
  metaHost[2] = int32(asP.cachedTokens)
  metaHost[3] = int32(asP.poolK.getImpl().shape()[2])
  for i in 0 ..< asP.pageSlots.len:
    metaHost[4 + i] = int32(asP.pageSlots[i])
  let metaDev = engine.allocDevice(int32, metaHost.len)
  engine.copyToDeviceFromHost(metaDev, metaHost)
  enqueuePoolWriteRows[E, Sh, ShS, Mem, typeof(metaDev)](engine, batch, asP, pk, pv, sk, sv, metaDev)
  # Output: the pool K slab, consumers read the written pool
  n = asP.poolK
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("pool_write", enqueuePoolWrite)
