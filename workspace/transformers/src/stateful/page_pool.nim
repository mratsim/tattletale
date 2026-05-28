## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  ./kvcache  # ceilDiv, round_step_down, round_step_up, TokensPerPage

type
  PagePool* = ref object
    ## Persistent stack GPU page allocator (value type for =destroy hook).
    k_buffer: Tensor         # (num_pages, num_layers, PAGE_SIZE, kv_heads, head_dim)
    v_buffer: Tensor         # same shape
    free_indices: seq[int32] # stack of available slot indices

  Page* = ref PageObj
  PageObj = object
    ## Underlying data for Page ref objects.
    index: int32 = -1i32
    pool {.cursor.}: PagePool # Back-pointer (orchestrator keeps pool alive, and we only have integers to return so don't refcount)
    k_view*: Tensor           # (num_layers, PAGE_SIZE, kv_heads, head_dim) — view into pool
    v_view*: Tensor           # same

# No custom =destroy, =copy, =sink hooks — ORC default field-by-field
# destruction handles Tensor/TorchTensor lifecycle correctly for PagePool
#
# For the Page, if the PagePool is not destroyed return the index to it so it can be borrowed again
# We need to nil Tensor field to ensure their destructor is called so the TorchTensor's refcount is decremnented and they are collected

proc `=destroy`(p: var PageObj) =
  ## Auto-recycle slot index to the pool's free stack.
  ## Called by ORC when the last Page ref is dropped.
  ## SAFETY: PagePool outlives all Pages (orchestrator lifetime).
  if p.pool != nil and p.index >= 0:
    p.pool.free_indices.add(p.index)
    p.index = -1
  p.k_view = nil
  p.v_view = nil

proc init*(_: type PagePool;
            num_pages: int; num_layers: int;
            kv_heads, head_dim: int;
            dtype: ScalarKind; device: DeviceKind): PagePool =
  let opts = F.tensorOptions(dtype, device)
  var free = newSeq[int32](num_pages)
  for i in 0 ..< num_pages:
    free[i] = (num_pages - 1 - i).int32
  result = PagePool(
    k_buffer: F.zeros(num_pages, num_layers, TokensPerPage, kv_heads, head_dim, opts),
    v_buffer: F.zeros(num_pages, num_layers, TokensPerPage, kv_heads, head_dim, opts),
    free_indices: free
  )

proc borrow*(pool: PagePool): Page =
  if pool.free_indices.len == 0:
    raise newException(ResourceExhaustedError,
      "[ttt] PagePool exhausted!")
  let idx = pool.free_indices.pop()
  result = Page(
    index: idx,
    pool: pool,
    k_view: pool.k_buffer[idx.int],
    v_view: pool.v_buffer[idx.int],
  )

func pagesAvailable*(pool: PagePool): int =
  pool.free_indices.len

func pageIndex*(p: Page): int32 =
  p.index
