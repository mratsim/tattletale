# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/tensorcery/src/tensor_datatypes,
  ./kvcache  # ceilDiv, round_step_down, round_step_up, TokensPerPage

type
  PagePool* = ref object
    ## Persistent stack GPU page allocator (value type, =destroy returns slots).
    ## K and V buffers take independent per-buffer widths. MLA pages put
    ## the compressed latent in the K buffer (k = kv_lora_rank) and the kpe
    ## plane in the V buffer (v = qk_rope_head_dim), kv_heads 1.
    k_buffer: Tensor  # (num_pages, num_layers, PAGE_SIZE, k_kv_heads, k_head_dim)
    v_buffer: Tensor  # (num_pages, num_layers, PAGE_SIZE, v_kv_heads, v_head_dim)
    free_indices: seq[int32] # stack of available slot indices

  Page* = ref PageObj
  PageObj = object
    ## Underlying data for Page ref objects.
    index: int32 = -1'i32
    pool {.cursor.}: PagePool # Back-pointer: the orchestrator keeps the pool alive,
    # and we only have integers to return, so don't refcount
    k_view*: Tensor           # (num_layers, PAGE_SIZE, k_kv_heads, k_head_dim) slab view
    v_view*: Tensor           # (num_layers, PAGE_SIZE, v_kv_heads, v_head_dim) slab view

# No custom =destroy, =copy, =sink operators. ORC default field-by-field
# destruction handles the Tensor lifecycle correctly for PagePool.
#
# Page slots return to the pool's free stack while the pool is still alive,
# then the tensor fields clear so their destructors run and release buffer refcounts.

proc `=destroy`(p: var PageObj) =
  ## Returns the slot index to the pool's free stack when the last Page ref drops.
  ## SAFETY: PagePool outlives all Pages (orchestrator lifetime).
  if p.pool != nil and p.index >= 0:
    p.pool.free_indices.add(p.index)
    p.index = -1'i32
  p.k_view = default(Tensor)
  p.v_view = default(Tensor)

proc init*(_: type PagePool, engine: AnyEngine,
            num_pages: int, num_layers: int,
            k_kv_heads, k_head_dim: int,
            v_kv_heads, v_head_dim: int): PagePool =
  ## Per-buffer widths: MLA calls this with k_kv_heads 1,
  ## k_head_dim kv_lora_rank, v_kv_heads 1, v_head_dim qk_rope_head_dim.
  ##
  ## Args:
  ## - engine: the engine that allocates the K and V slabs
  ## - num_pages, num_layers: pool capacity and layer count
  ## - k_kv_heads, k_head_dim: K buffer width
  ## - v_kv_heads, v_head_dim: V buffer width
  ##
  ## The buffers allocate over DeviceMem in bfloat16, the paged kernels'
  ## element dtype. Pages are written before they are read: the jagged attention
  ## kernel bounds every walk at a segment's covered tokens, so the allocation
  ## does no zero fill.
  var free = newSeq[int32](num_pages)
  for i in 0 ..< num_pages:
    free[i] = (num_pages - 1 - i).int32
  let
    kShape = @[num_pages, num_layers, TokensPerPage, k_kv_heads, k_head_dim]
    vShape = @[num_pages, num_layers, TokensPerPage, v_kv_heads, v_head_dim]
  result = PagePool(
    k_buffer: newTensor(engine, kShape, kBf16),
    v_buffer: newTensor(engine, vShape, kBf16),
    free_indices: free
  )

proc init*(_: type PagePool, engine: AnyEngine, num_pages: int, num_layers: int, kv_heads, head_dim: int): PagePool =
  ## Shared-width overload for the GQA page class.
  ## Both buffers take one (kv_heads, head_dim) shape.
  PagePool.init(engine, num_pages, num_layers,
    kv_heads, head_dim, kv_heads, head_dim)

proc spanToSeq(sp: Span[int]): seq[int] =
  ## Span to seq, the `view` face takes openArray[int]
  result = newSeq[int](sp.len)
  for i in 0 ..< sp.len:
    result[i] = sp[i]

proc borrow*(pool: PagePool): Page =
  if pool.free_indices.len == 0:
    raise newException(ResourceExhaustedError,
      "[ttt] PagePool exhausted!")
  let idx = pool.free_indices.pop()
  # Page slots build as shared-buffer layout views over the slab's first axis,
  # one (num_layers, PAGE_SIZE, kv_heads, head_dim) window per page,
  # no allocation and no copy.
  proc pageView(slab: Tensor, idx: int): Tensor =
    let impl = slab.node.getImpl()
    let s = impl.shape()
    let st = impl.strides()
    var shape = newSeq[int](s.len - 1)
    var strides = newSeq[int](st.len - 1)
    for i in 0 ..< shape.len:
      shape[i] = s[i + 1]
      strides[i] = st[i + 1]
    slab.view(shape, strides, impl.offset() + idx * st[0])
  result = Page(
    index: idx,
    pool: pool,
    k_view: pageView(pool.k_buffer, idx.int),
    v_view: pageView(pool.v_buffer, idx.int),
  )

func kBuffer*(pool: PagePool): Tensor =
  ## K slab of shape (num_pages, num_layers, PAGE_SIZE, k_kv_heads, k_head_dim).
  pool.k_buffer

func vBuffer*(pool: PagePool): Tensor =
  ## V slab of shape (num_pages, num_layers, PAGE_SIZE, v_kv_heads, v_head_dim).
  pool.v_buffer

func pagesAvailable*(pool: PagePool): int =
  pool.free_indices.len

func pageIndex*(p: Page): int32 =
  p.index

proc layerView*(pool: PagePool, layer: int): tuple[kView, vView: Tensor] =
  ## Per-layer slab views, the slice the paged kernels consume per dispatch.
  ##
  ## Returns:
  ## - kView, vView: (num_pages, PAGE_SIZE, kv_heads, head_dim) windows at the layer base
  ##   with page stride num_layers·PAGE_SIZE·kv_heads·head_dim
  ##
  ## The kernel `layer` arg must be 0 when a layerView view is passed,
  ## the offset is already applied.
  ##
  # Reference path: narrow axis 1 to one layer, squeeze the size-1 axis.
  # One shared-buffer layout view builds the same window: same strides,
  # element offset at the layer base.
  proc layerSlice(slab: Tensor, layer: int): Tensor =
    let impl = slab.node.getImpl()
    let s = impl.shape()
    let st = impl.strides()
    var shape = newSeq[int](s.len - 1)
    var strides = newSeq[int](st.len - 1)
    shape[0] = s[0]
    strides[0] = st[0]
    for i in 2 ..< s.len:
      shape[i - 1] = s[i]
      strides[i - 1] = st[i]
    slab.view(shape, strides, impl.offset() + layer * st[1])
  (layerSlice(pool.k_buffer, layer), layerSlice(pool.v_buffer, layer))
