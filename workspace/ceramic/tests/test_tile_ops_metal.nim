## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Host-side value checks of the tile op surface over the 8×8×8 FMA
## atom: the fp32 load/store dataflow and the op call semantics.
## Host-only: the fp16 path and the mma shuffle need the device.
##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/test_tile_ops_metal \
##   --nimcache:nimcache/tests/test_tile_ops_metal \
##   workspace/ceramic/tests/test_tile_ops_metal.nim

import std/strformat
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_configgen
import workspace/ceramic/src/hardware/h_registry
import workspace/ceramic/src/hardware/h_properties
import workspace/ceramic/src/tile_algebra/tiles
import workspace/ceramic/src/tile_algebra/tile_config
import workspace/ceramic/src/tile_algebra/tile_io
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/tile_algebra/tile_mma
import workspace/ceramic/src/tile_algebra/tile_epilogues
import workspace/ceramic/src/tile_algebra/tile_epilogues_backend

# ═════════════════════════════════════════════════════════════════════════
#  The fp32 load/store dataflow, end to end on the host (lane 0)
# ═════════════════════════════════════════════════════════════════════════
#
#  The host runs lane 0, whose fragment cell is (row, col) = (0, 0) and
#  whose (1, 1, 1) thread layout owns every subtile: the loads fill both
#  fragment values of every subtile, the stores write them all back.

proc checkFp32LoadStore() =
  var buf: array[32 * 16, float32]
  for i in 0 ..< buf.len:
    buf[i] = float32(i)
  let gdBuf = gd(cast[ptr UncheckedArray[float32]](addr buf[0]), 0, 0, 32, 16)
  var t: rt_l(float32, 32, 16, UNIVERSAL_8x8x8_F32F32F32F32)
  t.loadTile(gdBuf, (0'i32, 0'i32, 0'i32, 0'i32))
  # Lane 0 reads subtile (n, m)'s cells (8n, 8m) and (8n, 8m+1):
  # buffer elements (8n)·16 + 8m and +1.
  for n in 0 ..< 4:
    for m in 0 ..< 2:
      doAssert t.frags[n][m].frag[0] == float32(n * 128 + m * 8),
        &"load cell ({n},{m},0): {t.frags[n][m].frag[0]} != {float32(n * 128 + m * 8)}"
      doAssert t.frags[n][m].frag[1] == float32(n * 128 + m * 8 + 1),
        &"load cell ({n},{m},1): {t.frags[n][m].frag[1]} != {float32(n * 128 + m * 8 + 1)}"
  # The store mirrors the load: every subtile cell lands back at its
  # buffer element.
  var outBuf: array[32 * 16, float32]
  for i in 0 ..< outBuf.len:
    outBuf[i] = -1.0'f32
  let gdOutBuf = gd(cast[ptr UncheckedArray[float32]](addr outBuf[0]), 0, 0, 32, 16)
  gdOutBuf.storeTile(t, (0'i32, 0'i32, 0'i32, 0'i32))
  for n in 0 ..< 4:
    for m in 0 ..< 2:
      doAssert outBuf[n * 128 + m * 8] == float32(n * 128 + m * 8),
        &"store cell ({n},{m},0) mismatch"
      doAssert outBuf[n * 128 + m * 8 + 1] == float32(n * 128 + m * 8 + 1),
        &"store cell ({n},{m},1) mismatch"
  echo "  OK: fp32 load/store end to end (lane-0 fragment cells)"

proc checkFmaStorage() =
  # The 8×8×8 FMA atom addresses its storage partition with the atom's
  # subtile strides: the tile-plane strides come from the atom's M/N.
  # A 32×16 FMA tile holds 16 slots (rowTiles=4, colTiles=2, vpt=2).
  var buf: array[32 * 16, float32]
  for i in 0 ..< buf.len:
    buf[i] = float32(i)
  let gdBuf = gd(cast[ptr UncheckedArray[float32]](addr buf[0]), 0, 0, 32, 16)
  # Stride probe: the local_tile_dyn plane keeps the view's (rows, cols)
  # strides (32·strideRow, 16·strideCol); the subtile steps come from the
  # lane's fragment cell, evaluated from the atom's A/C layout at the load.
  let fmaView = local_tile_dyn(gdBuf, 32, 16, (0, 0, 0, 0))
  doAssert fmaView.layout.stride[0] == gdBuf.layout.stride[2],
    "the FMA tile-plane row stride must be the view's row stride"
  doAssert fmaView.layout.stride[1] == gdBuf.layout.stride[3],
    "the FMA tile-plane col stride must be the view's col stride"
  # Host load over lane 0's owned cells: slot (n, m, v) reads buffer
  # element (8n)·16 + 8m + v.
  var t: rt_l(float32, 32, 16, UNIVERSAL_8x8x8_F32F32F32F32)
  t.loadTile(gdBuf, (0'i32, 0'i32, 0'i32, 0'i32))
  for n in 0 ..< 4:
    for m in 0 ..< 2:
      doAssert t.frags[n][m].frag[0] == float32(n * 128 + m * 8),
        &"FMA load cell ({n},{m},0): {t.frags[n][m].frag[0]} != {float32(n * 128 + m * 8)}"
  # Host store round-trip over the owned cells.
  var outBuf: array[32 * 16, float32]
  for i in 0 ..< outBuf.len:
    outBuf[i] = -1.0'f32
  let gdOutBuf = gd(cast[ptr UncheckedArray[float32]](addr outBuf[0]), 0, 0, 32, 16)
  gdOutBuf.storeTile(t, (0'i32, 0'i32, 0'i32, 0'i32))
  for n in 0 ..< 4:
    for m in 0 ..< 2:
      doAssert outBuf[n * 128 + m * 8] == float32(n * 128 + m * 8),
        &"FMA store cell ({n},{m},0): {outBuf[n * 128 + m * 8]} != {float32(n * 128 + m * 8)}"
  echo "  OK: FMA atom storage (M×N 8×8, vpt=2) partitions lane 0's cells"

proc checkFmaMaps() =
  # The maps resolve explicit-atom tiles through the method-call
  # inference of the per-tile atom params (d.mul(a, b)); the map
  # iterates lane 0's owned subtiles (all of them) over both fragment
  # values.
  var fa: rt_l(float32, 8, 8, UNIVERSAL_8x8x8_F32F32F32F32)
  var fb: rt_l(float32, 8, 8, UNIVERSAL_8x8x8_F32F32F32F32)
  var fd: rt_l(float32, 8, 8, UNIVERSAL_8x8x8_F32F32F32F32)
  fa.frags[0][0].frag[0] = 1.0'f32
  fa.frags[0][0].frag[1] = 2.0'f32
  fb.frags[0][0].frag[0] = 3.0'f32
  fb.frags[0][0].frag[1] = 4.0'f32
  fd.mul(fa, fb)
  doAssert fd.frags[0][0].frag[0] == 3.0'f32 and fd.frags[0][0].frag[1] == 8.0'f32,
    "mul cell mismatch"
  # The row maps and the col-vec ops resolve the same way. The col-vec
  # of an 8-row FMA tile holds rowTiles = 1 value per fragment column
  # slot (vpt = 2).
  var rv: Tensor[float32, (Int[1], Int[2]), (Int[2], Int[1])]
  rv.data[0] = 10.0'f32
  rv.data[1] = 20.0'f32
  fd.mul_row(fa, rv)
  doAssert fd.frags[0][0].frag[0] == 10.0'f32 and fd.frags[0][0].frag[1] == 40.0'f32,
    "mul_row cell mismatch"
  fd.sub_row(fa, rv)
  doAssert fd.frags[0][0].frag[0] == -9.0'f32 and fd.frags[0][0].frag[1] == -18.0'f32,
    "sub_row cell mismatch"
  fd.div_row(fa, rv)
  doAssert fd.frags[0][0].frag[0] == 0.1'f32 and fd.frags[0][0].frag[1] == 0.1'f32,
    "div_row cell mismatch"
  echo "  OK: maps resolve explicit-atom tiles (method-call inference)"

# ═════════════════════════════════════════════════════════════════════════
#  The layer ops: method-call syntax, in-place forms allowed
# ═════════════════════════════════════════════════════════════════════════

proc checkMulTile() =
  var a: rt_l(float32, 8, 8, UNIVERSAL_8x8x8_F32F32F32F32)
  var b: rt_l(float32, 8, 8, UNIVERSAL_8x8x8_F32F32F32F32)
  var d: rt_l(float32, 8, 8, UNIVERSAL_8x8x8_F32F32F32F32)
  a.frags[0][0].frag[0] = 1.0'f32
  a.frags[0][0].frag[1] = 2.0'f32
  b.frags[0][0].frag[0] = 3.0'f32
  b.frags[0][0].frag[1] = 4.0'f32
  # explicit destination: d = a·b, operands unchanged
  d.mul(a, b)
  doAssert d.frags[0][0].frag[0] == 3.0'f32 and d.frags[0][0].frag[1] == 8.0'f32,
    "mul cell mismatch"
  doAssert a.frags[0][0].frag[0] == 1.0'f32 and b.frags[0][0].frag[1] == 4.0'f32,
    "mul must not modify the operands"
  # in-place: the mul result replaces the first operand
  a.mul(a, a)
  doAssert a.frags[0][0].frag[0] == 1.0'f32 and a.frags[0][0].frag[1] == 4.0'f32,
    "in-place mul cell mismatch"
  # the transposed-B map overload shares the same per-slot contract, with the
  # col-subtile-outer fragment grid (frags[m][n]); lane 0 owns every
  # subtile.
  var kr: rt_r(float32, 8, 16, UNIVERSAL_8x8x8_F32F32F32F32)
  var ks: rt_r(float32, 8, 16, UNIVERSAL_8x8x8_F32F32F32F32)
  var kd: rt_r(float32, 8, 16, UNIVERSAL_8x8x8_F32F32F32F32)
  for m in 0 ..< 2:
    for v in 0 ..< 2:
      kr.frags[m][0].frag[v] = float32(m * 8 + v + 3)
      ks.frags[m][0].frag[v] = float32(m * 8 + v + 4)
  kd.mul(kr, ks)
  for m in 0 ..< 2:
    for v in 0 ..< 2:
      doAssert kd.frags[m][0].frag[v] == kr.frags[m][0].frag[v] * ks.frags[m][0].frag[v],
        &"rt_r mul cell ({m},{v}) mismatch"
  echo "  OK: mul (tile × tile): method-call + in-place, both maps"

proc checkVecOps() =
  var v: Tensor[float32, (Int[1], Int[1]), (Int[1], Int[1])]
  v.data[0] = 1.0'f32
  # in-place scalar ops: v = (v·2) + 1
  v.mul(v, 2.0'f32)
  v.add(v, 1.0'f32)
  doAssert v.data[0] == 3.0'f32,
    "vec scalar op slot mismatch"
  # the Apple-width (vpt = 2) col-vec rides the same generic ops
  var v2: Tensor[float32, (Int[1], Int[2]), (Int[2], Int[1])]
  v2.data[0] = 1.0'f32
  v2.data[1] = 2.0'f32
  v2.mul(v2, 2.0'f32)
  v2.add(v2, 1.0'f32)
  doAssert v2.data[0] == 3.0'f32 and v2.data[1] == 5.0'f32,
    "vpt=2 col-vec scalar ops"
  echo "  OK: mul/add (col-vec × scalar): in-place forms, vpt-generic"

proc checkMulRow() =
  var a: rt_l(float32, 8, 16, UNIVERSAL_8x8x8_F32F32F32F32)
  var rv: Tensor[float32, (Int[1], Int[2]), (Int[2], Int[1])]
  var d: rt_l(float32, 8, 16, UNIVERSAL_8x8x8_F32F32F32F32)
  for m in 0 ..< 2:
    for v in 0 ..< 2:
      a.frags[0][m].frag[v] = float32(m * 8 + v + 1)
  rv.data[0] = 10.0'f32
  rv.data[1] = 20.0'f32
  d.mul_row(a, rv)
  for m in 0 ..< 2:
    for v in 0 ..< 2:
      let want = a.frags[0][m].frag[v] * rv.data[v]
      doAssert d.frags[0][m].frag[v] == want,
        &"mul_row cell ({m},{v}): {d.frags[0][m].frag[v]} != {want}"
  echo "  OK: mul_row (tile × col-vec row map)"

proc checkColVecPairOps() =
  var a: Tensor[float32, (Int[1], Int[2]), (Int[2], Int[1])]
  var b: Tensor[float32, (Int[1], Int[2]), (Int[2], Int[1])]
  var d: Tensor[float32, (Int[1], Int[2]), (Int[2], Int[1])]
  for i in 0 ..< a.data.len:
    a.data[i] = float32(i + 1)
    b.data[i] = float32(20 + i)
  # sub: d = b − a (the online-softmax m_prev − m step)
  d.sub(b, a)
  for i in 0 ..< d.data.len:
    doAssert d.data[i] == b.data[i] - a.data[i], &"sub slot {i} mismatch"
  # mul: d = a·b (the online-softmax l × exp2(m_prev − m) step)
  d.mul(a, b)
  for i in 0 ..< d.data.len:
    doAssert d.data[i] == a.data[i] * b.data[i], &"mul vec slot {i} mismatch"
  # copy: d = a
  d.copy(a)
  for i in 0 ..< d.data.len:
    doAssert d.data[i] == a.data[i], &"copy slot {i} mismatch"
  # zero: d = 0 (the running-sum seed)
  d.zero()
  for i in 0 ..< d.data.len:
    doAssert d.data[i] == 0.0'f32, &"zero slot {i} mismatch"
  # neg_infty: the most-negative finite seed (the running-max seed)
  d.neg_infty()
  for i in 0 ..< d.data.len:
    doAssert d.data[i] == -3.402823466e38'f32, &"neg_infty slot {i} mismatch"
  echo "  OK: col-vec pair ops (sub/mul/copy/zero/neg_infty)"

proc checkRowMaps() =
  var a: rt_l(float32, 8, 16, UNIVERSAL_8x8x8_F32F32F32F32)
  var rv: Tensor[float32, (Int[1], Int[2]), (Int[2], Int[1])]
  var d: rt_l(float32, 8, 16, UNIVERSAL_8x8x8_F32F32F32F32)
  for m in 0 ..< 2:
    for v in 0 ..< 2:
      a.frags[0][m].frag[v] = float32(m * 8 + v + 1)
  rv.data[0] = 10.0'f32
  rv.data[1] = 20.0'f32
  d.sub_row(a, rv)
  for m in 0 ..< 2:
    for v in 0 ..< 2:
      let want = a.frags[0][m].frag[v] - rv.data[v]
      doAssert d.frags[0][m].frag[v] == want,
        &"sub_row cell ({m},{v}): {d.frags[0][m].frag[v]} != {want}"
  d.div_row(a, rv)
  for m in 0 ..< 2:
    for v in 0 ..< 2:
      let want = a.frags[0][m].frag[v] / rv.data[v]
      doAssert d.frags[0][m].frag[v] == want,
        &"div_row cell ({m},{v}): {d.frags[0][m].frag[v]} != {want}"
  echo "  OK: sub_row/div_row (tile − / ÷ col-vec row maps)"

proc checkRtRightStore() =
  # The store through the transposed-B view: lane 0 writes every
  # subtile's cells. Slot (m, n, v) lands at buffer offset
  # n·8 + m·128 + v·16 (the colTile tile-plane strides
  # (M·strideCol, N·strideRow, strideRow) = (8, 128, 16)).
  var kt: rt_r(float32, 8, 16, UNIVERSAL_8x8x8_F32F32F32F32)
  for m in 0 ..< 2:
    for v in 0 ..< 2:
      kt.frags[m][0].frag[v] = float32(m * 8 + v + 1)
  var outBuf: array[8 * 16, float32]
  for i in 0 ..< outBuf.len:
    outBuf[i] = -1.0'f32
  let gdOutBuf = gd(cast[ptr UncheckedArray[float32]](addr outBuf[0]), shape = (-1, -1, -1, -1), stride = (
                            8 * 16, 0, 1, 16))
  gdOutBuf.storeTile(kt, (0'i32, 0'i32, 0'i32, 0'i32))
  for m in 0 ..< 2:
    for v in 0 ..< 2:
      doAssert outBuf[m * 8 + v] == float32(m * 8 + v + 1),
        &"transposed-B store cell (m={m}, v={v}) mismatch"
  echo "  OK: rt_r fp32 transposed-B store host round trip (lane-0 cells)"

proc checkEpiApply() =
  # The fragment-resident epilogue applies run on the host over lane 0's
  # owned cells. The EpiAXPBY C operand comes from the shard macro over
  # the same tile, so the shardView base (the tile-origin offset plus the
  # lane's fragment cell) is exercised here too.
  var ab: rt_l(float32, 8, 8, UNIVERSAL_8x8x8_F32F32F32F32)
  ab.frags[0][0].frag[0] = -3.0'f32
  ab.frags[0][0].frag[1] = 2.0'f32
  var d: rt_l(float32, 8, 8, UNIVERSAL_8x8x8_F32F32F32F32)
  d.frags[0][0].frag[0] = -7.0'f32  # sentinel
  d.frags[0][0].frag[1] = -7.0'f32
  EpiReLU().apply(d, ab)
  doAssert d.frags[0][0].frag[0] == 0.0'f32 and d.frags[0][0].frag[1] == 2.0'f32,
    "relu owned cell mismatch"
  # EpiAXPBY through the kernel's shard path: the per-lane C view is
  # built over the 8×8 buffer at the tile origin (1, 2), so the shard
  # base lands at buffer element (1·8)·8 + (2·8)·1 = 80 and lane 0's
  # cells are cbuf[80] and cbuf[81].
  var cbuf: array[144, float32]
  for i in 0 ..< cbuf.len:
    cbuf[i] = float32(i)
  let cptr = cast[ptr UncheckedArray[float32]](addr cbuf[0])
  var o2 = shard(initEpiAXPBY(2'f32, 3'f32, cView(float32, 8, 8, cptr)),
                 cptr, (0'i32, 0'i32, 1'i32, 2'i32), d)
  o2.apply(d, ab)
  doAssert d.frags[0][0].frag[0] == 2'f32 * (-3.0'f32) + 3'f32 * 80.0'f32,
    "axpby owned cell 0 mismatch"
  doAssert d.frags[0][0].frag[1] == 2'f32 * 2.0'f32 + 3'f32 * 81.0'f32,
    "axpby owned cell 1 mismatch"
  echo "  OK: fragment-resident epilogue applies (lane-0 owned cells)"

# ═════════════════════════════════════════════════════════════════════════
#  mma_AB: device-only, compile-checked
# ═════════════════════════════════════════════════════════════════════════

# The shuffle mma reads the lane id and gathers other lanes' registers,
# which cannot run on the host. Its contract (products accumulate in k
# order 0,1,2,… like the CPU reference) is asserted by the on-device gemm
# check at |Δ| = 0.0. This probe checks that the fp32-A form instantiates
# inside a DSL block.
const mmaCompileProbe = metal:
  proc mmaProbe(d: ptr UncheckedArray[float32], a, b: ptr UncheckedArray[float16]) {.global.} =
    var a_rtl: rt_l(float16, 8, 8)
    var b_rtr: rt_r(float16, 8, 8)
    var d_rtl: rt_l(float32, 8, 8, getTileConfig(float32, float16))
    let gd_a = a.gd(0, 0, 8, 8)
    let gd_b = b.gd(shape = (-1, -1, -1, -1), stride = ( 0, 0, 1, 8))
    a_rtl.loadTile(gd_a, (0'i32, 0'i32, 0'i32, 0'i32))
    b_rtr.loadTile(gd_b, (0'i32, 0'i32, 0'i32, 0'i32))
    d_rtl.zero()
    d_rtl.mma_AB(a_rtl, b_rtr)
    let gd_d = d.gd(0, 0, 8, 8)
    gd_d.storeTile(d_rtl, (0'i32, 0'i32, 0'i32, 0'i32))

when isMainModule:
  echo "TILE OPS HOST PASS"
  checkFp32LoadStore()
  checkFmaStorage()
  checkFmaMaps()
  checkMulTile()
  checkVecOps()
  checkMulRow()
  checkColVecPairOps()
  checkRowMaps()
  checkRtRightStore()
  checkEpiApply()
  echo "ALL TILE OPS HOST CHECKS PASS"
