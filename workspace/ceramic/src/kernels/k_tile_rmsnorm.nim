# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensor RMSNorm as a device function over row strips.
##
## - `rmsnorm(dst, src, eps, axis)`: x * rsqrt(mean(x²) + eps) over the input axis
## - `rmsnorm(dst, src, weight, eps, axis)`: the same, scaled by the per-axis weight
##
## Launch the kernels with `launchRmsnorm`: the one-warp-per-row launch contract
## is checked loud on the host, before any device work.

import workspace/crucible
import workspace/ceramic/src/tile_algebra

func staticShape[T, Sh, St](_: typedesc[TensorView[T, Sh, St]]): Sh =
  ## Compile-time shape of a tensor type: the shape tuple of the view's layout.
  ##
  ## - value-level shape reads such as `weight.rank()` do not fold in device code,
  ##   the shape type does: the codegen takes the raw generic body
  ## Static leaves are `Int[V]` phantom values, their value compares through `===`.
  ## Runtime leaves are plain `int` and `default` reads 0.
  ##
  ## Example: `staticShape(typeof(view)).rank()` is the view's rank.
  default(Sh)

proc rmsnormImpl(dst, src, weight: distinct TensorView, eps: float32, axis: static int, withWeight: static bool) {.inline.} =
  ## Row-wise RMS normalization shared by the two entry points.
  ##
  ## One warp per row, a grid-stride loop covers the row list from any launch grid size.
  ## Each output row is x * rsqrt(mean(x²) + eps), times the per-axis weight when withWeight. dst is fully overwritten.
  ##
  ## Sum of squares per row, one register tile at a time:
  ## - `rowView` cuts the source and destination strips per row, the rank-1 weight strip is the full tensor
  ## - tile loads pad past-the-end slots with the `roSum` identity, so short and ragged rows contribute nothing to the sum
  ## - `reduce(acc, reg, roSum, clear = false)` accumulates each tile's sum of squares onto the accumulator, one subgroup butterfly per tile
  ##
  ## Mean accumulation in fp32. `rsqrt` is the backend native reciprocal square root.
  ##
  ## Preconditions:
  ## - the axis size is at least 1
  ## - dst has the same shape as src
  ## - the launch threadgroup holds exactly one warp's threads, `getLaneWidth()` in total.
  ##   `launchRmsnorm` rejects other launches loud on the host before any device work,
  ##   a raw `launchDeviceFn` launch comes back NaN
  ## - when withWeight, the weight is rank 1 with one slot per axis position, see `rmsnorm`
  ##
  ## Args:
  ## - dst: same shape as src, receives the normalized rows
  ## - src: the source tensor
  ## - weight: one slot per axis position, shape (axis size,), any strides,
  ##   unread when withWeight is false
  ##
  ## Scalars:
  ## - eps: the stability constant added to the mean of squares
  ## - axis: the axis normalized over
  ## - withWeight: static switch, false compiles the weight out
  var reg: tile(float32)
  const
    W = getLaneWidth()
    owned = reg.size() div W
  let thrID = getLocalThreadID()
  # NaN from a runtime division: a folded 0/0 constant emits an invalid MSL token
  var poison = zero(float32)
  poison = poison / poison
  # The reduce's butterfly is subgroup-scoped: a threadgroup of any other size leaves
  # per-simdgroup partials uncombined or rows partially written.
  # Deferred: cross-warp combine via shared memory and barrier.
  let
    nThreads = threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z
    offContract = nThreads != W
  when withWeight:
    const
      wShape = staticShape(typeof(weight))
      wLeaf = when wShape is tuple: wShape[0] else: wShape
      sShape = staticShape(typeof(src))
      sLeaf = when sShape is tuple: sShape[axis] else: sShape
      sizeIsStatic = typeof(wLeaf) is Int and typeof(sLeaf) is Int
    static:
      doAssert wShape.rank() == 1,
        "rmsnorm weight must be rank 1, one slot per axis position"
    # the weight size folds statically when both leaves are compile-time literals,
    # runtime-shaped weights are rejected on the device instead
    when sizeIsStatic:
      static:
        doAssert wLeaf === sLeaf,
          "rmsnorm weight size must match the source axis size"
    else:
      # runtime-shaped weight: its size is only known on the device.
      # A size miss poisons the rows below.
      let wSize = weight.size()
  let
    rows = src.sliceAxes(axis).size()
    gridStride = getGridSize()
  for g in countup(getLocalTileID(src.shape), rows - 1, gridStride):
    let
      srcRow = rowView(src, axis, g)
      dstRow = rowView(dst, axis, g)
      rowSize = srcRow.size()
      tiled = make_layout(rowSize).logical_divide(reg.getLayout())
      stripTiles = tiled.getTileGrid().size()
    # pass 1: sum of squares, one register tile at a time
    var acc = zero(float32)
    for tileID in 0 ..< stripTiles:
      reg.load(srcRow, thrID, tileID, padWith = reduceIdentity[float32](roSum))
      for i in 0 ..< owned:
        reg.data[i] = reg.data[i] * reg.data[i]
      reduce(acc, reg, roSum, clear = false)
    let scale = rsqrt(acc / float32(rowSize) + eps)
    # pass 2: scaled apply, the weight broadcast along the axis
    when withWeight:
      let weightStrip = rowView(weight, 0, 0)
      for r in countup(thrID, rowSize - 1, W):
        dstRow[r] = srcRow[r] * scale * weightStrip[r]
    else:
      for r in countup(thrID, rowSize - 1, W):
        dstRow[r] = srcRow[r] * scale

proc rmsnorm*(dst, src: distinct TensorView, eps: float32, axis: static int) {.inline.} =
  ## Root-mean-square normalization over the input axis, fully overwriting dst.
  ##
  ## Each row computes x * rsqrt(mean(x²) + eps).
  ## Mean accumulation in fp32, identical to the reference implementations.
  ##
  ## Preconditions:
  ## - the axis size is at least 1
  ## - dst has the same shape as src
  ## - the launch threadgroup holds exactly one warp's threads, `getLaneWidth()` in total
  ##
  ## Off-contract launches are rejected: `launchRmsnorm` fails loud on the host,
  ## a raw `launchDeviceFn` launch comes back all-NaN.
  ##
  ## Args:
  ## - dst: same shape as src, receives the normalized rows
  ## - src: the source tensor
  ## - eps: the stability constant added to the mean of squares
  ## - axis: the axis normalized over
  rmsnormImpl(dst, src, src, eps, axis, withWeight = false)

proc rmsnorm*(dst, src, weight: distinct TensorView, eps: float32, axis: static int) {.inline.} =
  ## Root-mean-square normalization over the input axis, per-axis weight,
  ## fully overwriting dst.
  ##
  ## Each row computes x * rsqrt(mean(x²) + eps) * weight, the weight broadcast
  ## along the normalized axis. Element r of the row scales by weight[r],
  ## as in `mx.fast.rms_norm(x, weight, eps)` on the last axis.
  ##
  ## Preconditions:
  ## - the axis size is at least 1
  ## - dst has the same shape as src
  ## - the launch threadgroup holds exactly one warp's threads, `getLaneWidth()` in total
  ## - the weight is rank 1 with one slot per axis position, shape (axis size,)
  ##
  ## Off-contract launches are rejected: `launchRmsnorm` fails loud on the host,
  ## a raw `launchDeviceFn` launch comes back all-NaN.
  ##
  ## A wrong weight fails to compile when its shape is a compile-time literal,
  ## otherwise the launch comes back all-NaN.
  ##
  ## Args:
  ## - dst: same shape as src, receives the normalized rows
  ## - src: the source tensor
  ## - weight: one slot per axis position, shape (axis size,), any strides
  ##
  ## Scalars:
  ## - eps: the stability constant added to the mean of squares
  ## - axis: the axis normalized over
  rmsnormImpl(dst, src, weight, eps, axis, withWeight = true)

template rmsnormLaneWidth(E: typedesc[HwEngine]): int =
  ## Lane width of the backend the engine drives, as a compile-time constant.
  ##
  ## Same value as tile_algebra's `getLaneWidth()` for the same backend.
  ## Evaluable on the host: `getLaneWidth()` resolves inside device codegen,
  ## the launch-site contract check runs before any device work.
  ## The two tables cover the same backends and must change together.
  when E is CudaEngine:
    32
  elif E is MetalEngine:
    32
  elif E is OpenCLEngine:
    32
  elif E is VulkanEngine:
    32
  elif E is WebgpuEngine:
    32

template checkRmsnormLaunchBlk[E: HwEngine](engine: E, launchBlk: tuple) =
  ## One-warp-per-row launch contract, checked on the host at the launch site.
  ##
  ## The kernel's butterfly is simdgroup-scoped: the threadgroup must hold exactly
  ## one warp's threads, `rmsnormLaneWidth(E)` in total. Any other size raises
  ## before any device work.
  ##
  ## The kernel's device-side NaN guard stays as the last line of defense,
  ## covering launches that bypass `launchRmsnorm`.
  const laneW = rmsnormLaneWidth(E)
  let threads = dim3(launchBlk)
  doAssert threads.x * threads.y * threads.z == laneW,
    "rmsnorm launch: the threadgroup must hold exactly one warp's threads, " &
    $laneW & " lanes in total, got " & $threads.x & "x" & $threads.y & "x" & $threads.z

template launchRmsnorm*[E: HwEngine](engine: E, dst, src: auto, eps: float32, axis: static int, launchGrid: tuple, launchBlk: tuple) =
  ## Launches the unweighted rmsnorm, the one-warp-per-row contract checked on the host.
  ##
  ## The threadgroup must hold exactly one warp's threads, `getLaneWidth()` in total.
  ## Any other size raises at the launch site, before any device work.
  ##
  ## A raw `launchDeviceFn` bypass skips this check, the kernel's device-side
  ## NaN guard rejects it instead.
  ##
  ## Example:
  ##   engine.launchRmsnorm(dstDev, srcDev, 1e-5'f32, axis, (2, 1), (32, 1))
  ##
  ## Args:
  ## - dst, src: the tensors normalized in place over axis, dst same shape as src
  ## - launchGrid, launchBlk: the launch grid and threadgroup, any tuple dim3 accepts,
  ##   the flat threadgroup size must be one warp
  ##
  ## Scalars:
  ## - eps: the stability constant added to the mean of squares
  ## - axis: the axis normalized over
  checkRmsnormLaunchBlk(engine, launchBlk)
  engine.launchDeviceFn rmsnorm(dst, src, eps, axis) do:
    grid = launchGrid
    blk = launchBlk

template launchRmsnorm*[E: HwEngine](engine: E, dst, src, weight: auto, eps: float32, axis: static int, launchGrid: tuple, launchBlk: tuple) =
  ## Launches the weighted rmsnorm, the one-warp-per-row contract checked on the host.
  ##
  ## The threadgroup must hold exactly one warp's threads, `getLaneWidth()` in total.
  ## Any other size raises at the launch site, before any device work.
  ##
  ## A raw `launchDeviceFn` bypass skips this check, the kernel's device-side
  ## NaN guard rejects it instead.
  ##
  ## Example:
  ##   engine.launchRmsnorm(dstDev, srcDev, weightDev, 1e-5'f32, axis, (2, 1), (32, 1))
  ##
  ## Args:
  ## - dst, src: the tensors normalized in place over axis, dst same shape as src
  ## - weight: one slot per axis position, shape (axis size,), any strides
  ## - launchGrid, launchBlk: the launch grid and threadgroup, any tuple dim3 accepts,
  ##   the flat threadgroup size must be one warp
  ##
  ## Scalars:
  ## - eps: the stability constant added to the mean of squares
  ## - axis: the axis normalized over
  checkRmsnormLaunchBlk(engine, launchBlk)
  engine.launchDeviceFn rmsnorm(dst, src, weight, eps, axis) do:
    grid = launchGrid
    blk = launchBlk
