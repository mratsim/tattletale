# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Tile load/store over a TensorBasicView: a strided register tile map
## pads the out-of-bounds positions on load and drops them on store
## Run:
##   nim c -r -d:release --hints:off --warnings:off --outdir:build/tests/t_basic_views_tile_bounds \
##     --nimcache:nimcache/t_basic_views_tile_bounds workspace/tensorcery/tests/t_basic_views_tile_bounds.nim

import workspace/crucible
import workspace/ceramic
import workspace/tensorcery/src/tensors_internals

proc main =
  # Tile map (4, 4):(1, 4): thread t owns the positions t, t+4, t+8, t+12 of a 16-element tile.
  # Tile 1 of a 20-element view maps onto the positions 16, 20, 24, 28: three exceed the view length.
  # Buffer size 32 keeps the 12 entries after the view length as poison -999, an out-of-bounds access reads poison.
  const
    viewLen = 20
    poison = -999.0'f32
    pad = 777.0'f32

  # Tile 1, thread 0: positions 16, 20, 24, 28
  block: # load pads the past-the-view positions with `padWith`
    var buf: array[32, float32]
    for i in 0 ..< 32:
      buf[i] = if i < viewLen: i.float32 else: poison
    var src: TensorBasicView[float32]
    src.data = cast[ptr UncheckedArray[float32]](addr buf[0])
    src.layout = makeLayoutBasic([viewLen])
    let reg = tile(float32, make_layout((4, 4), (1, 4))).load(src, 0, 1, pad)
    doAssert reg.data[0] == 16.0'f32, "position 16 is in bounds: " & $reg.data[0]
    doAssert reg.data[1] == pad, "position 20 pads, got " & $reg.data[1]
    doAssert reg.data[2] == pad, "position 24 pads, got " & $reg.data[2]
    doAssert reg.data[3] == pad, "position 28 pads, got " & $reg.data[3]

  # Tile 0, thread 0: positions 0, 4, 8, 12, all in bounds
  block: # load keeps the in-bounds fast path
    var buf: array[32, float32]
    for i in 0 ..< 32:
      buf[i] = i.float32
    var src: TensorBasicView[float32]
    src.data = cast[ptr UncheckedArray[float32]](addr buf[0])
    src.layout = makeLayoutBasic([viewLen])
    let reg = tile(float32, make_layout((4, 4), (1, 4))).load(src, 0, 0, pad)
    doAssert reg.data[0] == 0.0'f32
    doAssert reg.data[1] == 4.0'f32
    doAssert reg.data[2] == 8.0'f32
    doAssert reg.data[3] == 12.0'f32

  block: # store drops the past-the-view positions
    var buf: array[32, float32]
    for i in 0 ..< 32:
      buf[i] = if i < viewLen: i.float32 else: poison
    var dst: TensorBasicView[float32]
    dst.data = cast[ptr UncheckedArray[float32]](addr buf[0])
    dst.layout = makeLayoutBasic([viewLen])
    var reg: tile(float32, make_layout((4, 4), (1, 4)))
    # This thread's register slots hold 101, 102, 103, 104 for positions 16, 20, 24, 28
    reg.data[0] = 101.0'f32
    reg.data[1] = 102.0'f32
    reg.data[2] = 103.0'f32
    reg.data[3] = 104.0'f32
    dst.store(reg, 0, 1)
    doAssert buf[16] == 101.0'f32, "position 16 stores: " & $buf[16]
    doAssert buf[20] == poison, "position 20 drops, got " & $buf[20]
    doAssert buf[24] == poison, "position 24 drops, got " & $buf[24]
    doAssert buf[28] == poison, "position 28 drops, got " & $buf[28]

  echo "tensorcery basic-view tile bounds (host): all checks passed"

when isMainModule:
  main()
