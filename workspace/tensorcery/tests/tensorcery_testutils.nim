# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import std/random
import workspace/tensorcery

export tensorcery

proc randomSeq*[T](rng: var Rand, _: typedesc[T], size: int, bounds: HSlice[float32, float32] = 1.0'f32 .. 2.0'f32): seq[T] =
  result.newSeq(size)
  for i in 0 ..< size:
    when T is float32:
      result[i] = rng.rand(bounds)
    else:
      result[i] = rng.rand(bounds.a .. bounds.b).to(T)

proc randomTensor*[T](rng: var Rand, _: typedesc[T], shape: varargs[int], bounds: HSlice[float32, float32] = 1.0'f32 .. 2.0'f32): Tensor =
  var size = 1
  for dim in shape:
    size *= dim
  rng.randomSeq(T, size, bounds).toTensor(shape)
