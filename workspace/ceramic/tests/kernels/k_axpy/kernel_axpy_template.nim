# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[math, random, strformat, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/src/kernels/k_tile_axpy

export int_tuples
export layout_algebra
export tensors
export k_tile_axpy

export int_tuples, int_tuples_streams
export k_tile_axpy

# ═══════════════════════════════════════════════════════════════
#  fp32 host reference
# ═══════════════════════════════════════════════════════════════

func refAxpy[T, ShX, StX, ShY, StY, ShR, StR](
    alpha: float32, x: TensorView[T, ShX, StX], y: TensorView[T, ShY, StY],
    res: TensorView[T, ShR, StR]): void =
  ## out = alpha*x + y on the host, through each tensor's own layout
  for r in 0 ..< y.shape[0]:
    for c in 0 ..< y.shape[1]:
      res[r, c] = alpha * x[r, c] + y[r, c]

# ═══════════════════════════════════════════════════════════════
#  The checks
# ═══════════════════════════════════════════════════════════════

proc checkAxpy*[E: HwEngine](
    rng: var Rand, engine: E, rows, cols: static int,
    sX, sY: static (int, int), msg: static string,
    lang: static string = "spirv") =
  ## One axpy launch over random (rows, cols) tensors with the given strides,
  ## the result compared against the host reference.
  ## `lang` selects the Vulkan SPIR-V ingestion language, words route by default.
  var x = rng.randomHostTensor( float32, (rows, cols), sX)
  var y = rng.randomHostTensor( float32, (rows, cols), sY)
  var expect = makeHostTensor(float32, (rows, cols), (cols, 1))
  refAxpy(2.5'f32, x.view(), y.view(), expect.view())
  let
    xDev = engine.bridge(x)
    yDev = engine.bridge(y)
  # the kernel's grid-stride loop completes the tile grid from any launch grid
  let gridX = rng.rand(1 .. 2 * ceil_div(rows * cols div 128, 32) + 1)
  template launch(langLit: untyped) =
    engine.launchDeviceFn axpy(yDev, xDev, 2.5'f32) do:
      grid = (gridX, 1)
      blk = (32, 1)
      lang = `langLit`
  when lang == "spirv":
    launch("spirv")
  elif lang == "glsl":
    launch("glsl")
  else:
    launch(hlsl)
  engine.copyToHostFromDevice(addr y.data[0], yDev.data, y.layout.cosize())
  assertAllClose(y.view(), expect.view(), rtol = 1e-5, abstol = 1e-4, msg = msg)

# ═══════════════════════════════════════════════════════════════
#  The drivers
# ═══════════════════════════════════════════════════════════════

proc run_axpy_tests*[E: HwEngine](engine: E, moduleName: string, lang: static string = "spirv") =
  var rng: Rand
  let seed = uint32(getTime().toUnix() and (1'i64 shl 32 - 1)) # unixTime mod 2^32
  rng.seed(int64(seed))
  echo "\n------------------------------------------------------\n"
  echo moduleName, " xoroshiro128+ seed: ", seed

  const rows = 8
  const cols = 128

  rng.checkAxpy(engine, rows, cols, (cols, 1), (cols, 1), "row-major", lang)
  rng.checkAxpy(engine, rows, cols, (1, rows), (1, rows), "col-major", lang)
  rng.checkAxpy(engine, rows, cols, (2*cols, 2), (2*cols, 2), "stride-2", lang)
  rng.checkAxpy(engine, rows, cols, (1, cols), (1, cols), "inverted strides", lang)
  rng.checkAxpy(engine, rows, cols, (cols, 1), (1, rows), "row-major x, col-major y", lang)
