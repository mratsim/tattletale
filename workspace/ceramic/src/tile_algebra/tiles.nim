# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/typetraits
import workspace/crucible
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

# ############################################################
#
#                      Tiles
#
# ############################################################

# ═══════════════════════════════════════════════════════════════
#  Backend magic functions
# ═══════════════════════════════════════════════════════════════

template getLaneWidth*: int =
  ## Returns the width of the lanes of one co-executing group
  ## as a compile-time constant, assuming the reference hardware.
  ## Pass the tile map explicitly with `tile(T, make_layout(...))` on other devices.
  ## - CUDA: one warp, 32 lanes
  ## - Metal: one simdgroup, 32 lanes
  ## - OpenCL or Vulkan (GLSL or HLSL leg): one subgroup, 32 lanes (NVIDIA warp, AMD wave32)
  ## - WebGPU: one subgroup, 32 lanes (typical subgroup min size)
  when ccGetBackend() == ctCuda: 32
  elif ccGetBackend() == ctMetal: 32
  elif ccGetBackend() == ctOpenCL: 32
  elif ccGetBackend() == ctGlsl:
    32
  elif ccGetBackend() == ctHlsl:
    32
  elif ccGetBackend() == ctWebGPU: 32

template getGridSize*: int =
  ## Returns the total number of threadgroups in the launch grid
  threadgroups_per_grid.x * threadgroups_per_grid.y * threadgroups_per_grid.z # gridsize-allow

template getDefaultTileLayout*(T: typedesc): untyped =
  ## Returns the default register tile map
  ##
  ## For float32 (4-bytes) over a warp size of 32 threads:
  ##
  ##   getDefaultTileLayout(float32) == make_layout((32, 4), (4, 1))
  ##   # mode 0 = thread index, mode 1 = element slot
  ##   # thread t owns elements t*4 .. t*4+3 of a 128-element block
  ##
  ## Explanation:
  ## - chunk = 16 div sizeof(T): one 16-byte vector load per thread
  ## - 32 contiguous chunks: the threadgroup's span stays coalesced
  ## - thread stride is chunk: one 16-byte vector load per thread,
  ##   an interleaved map costs chunk scalar loads per thread
  ## - a tile map is a thread->element assignment
  const chunk = max(1, 16 div sizeof(T))
  make_layout((getLaneWidth(), chunk), (chunk, 1))

# ═══════════════════════════════════════════════════════════════
#  Tile datatype
# ═══════════════════════════════════════════════════════════════

type Tile*[T, Sh, St] = TensorOwned[T, Sh, St]

macro tile*(T: typedesc, L: typed): typedesc =
  ## Register tile type from a layout, e.g. an atom's thread->value map.
  ##
  ## Contract:
  ## - the Tile over T with the layout's static shape and stride
  let ty = getTypeInst(L)
  doAssert ty.kind == nnkBracketExpr and ty.len == 3,
    "tile: expected a Layout[Shape, Stride] type, got " & ty.repr
  result = nnkBracketExpr.newTree(bindSym"Tile", T, ty[1], ty[2])

template tile*(T: typedesc): typedesc =
  ## Register tile with the default tile map
  tile(T, getDefaultTileLayout(T))

template tile*(T: typedesc, shape: tuple): typedesc =
  ## Register tile type from a shape tuple, compact LayoutLeft strides
  ## through make_layout: `tile(float32, (32, 32))`
  ## - strides are the LayoutLeft prefix products, first mode stride 1
  ## - tile maps stay composable without hand-written strides
  tile(T, make_layout(shape))

# ═══════════════════════════════════════════════════════════════
#  Tile geometry
# ═══════════════════════════════════════════════════════════════

template tileWith*(v: TensorView, tiler: static tuple): auto =
  ## Tile geometry of `v`: its layout zipped-divided by the tiler,
  ## ((tile shape), (tile grid)) mode pairs.
  v.getLayout().zipped_divide(tiler)

template getTileGrid*(geom: Layout): auto =
  ## Tile grid of the geometry, the tile coordinates covering the tensor.
  geom.shape[1]

# ═══════════════════════════════════════════════════════════════
#  Sizes and fill values
# ═══════════════════════════════════════════════════════════════

template zero*(T: typedesc): untyped =
  when T is float16: cast[T](0'u16)
  elif T is bfloat16: cast[T](0'u16)
  else: T(0)

# ═══════════════════════════════════════════════════════════════
#  Launch mapping
# ═══════════════════════════════════════════════════════════════

template getLocalTileID*(gridShape: IntOrIntTuple): auto =
  ## Squadron position mapped to a flat index over the tile grid
  ## through the fleet layout.
  let (w, h, d) = (threadgroups_per_grid.x, threadgroups_per_grid.y, threadgroups_per_grid.z)
  let (x, y, z) = (threadgroup_position_in_grid.x, threadgroup_position_in_grid.y, threadgroup_position_in_grid.z)
  let fleet = make_layout(makeIntTuple((w, h, d)))
  crd2idx(fleet, (x, y, z))

template getLocalThreadID*: int =
  ## Lane index of the thread inside its squadron.
  thread_index_in_threadgroup  # threadindex-allow
