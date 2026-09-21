# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Spike driver for one gated delta-rule decode step through the crucible IR
## path on Metal, compared elementwise against the naive `gdnDecodeStep`
## reference from `tests/naive/`.
##
##   S ← S·exp2(g·log2e) + k ⊗ (β·(v − (S·exp2(g·log2e))·k))    y ← S'·(q·Dk^-0.5)
##
## - Driver only, no `t_`/`test_` name and no test umbrella, the production kernel shape lands in `src/kernels/ceramic/sequence_mixers/`
## - One (bh, 8-row) state tile per threadgroup, all state arithmetic fp32, the family dtype only at the loads and at the one y rounding
## - Kept as the band model's validation record, the core above is superseded by `src/kernels/ceramic/sequence_mixers/state_space/gdn/`
##
## Both sides run the same fp32 state arithmetic, the spelling deltas are
## the decay form (`exp(g)` vs `exp2(g·log2e)`, ≤ 4·2⁻²⁴ relative), the q̃ scale
## (divide by √Dk vs rsqrt-multiply, ≤ 2·2⁻²¹ relative) and the dot order
##
## - Model stated before measurement and judged per element, u₃₂ = 2⁻²⁴ is the fp32 unit roundoff
##
## | symbol | value                         |
## | ------ | ----------------------------- |
## | a, b   | abs(S·exp(g)), abs(k·δ)       |
## | kvAbs  | Σ_dkc abs(S·k) over the row r |
## | δ      | β·(v − Σ_dkc S·k)             |
## | yAbs   | Σ_dkc abs(S'·q̃) over the row |
## | u_fam  | 2⁻⁸ for bf16, 2⁻¹¹ for fp16   |
##
## | bar                | bound                                                                                              |
## | ------------------ | -------------------------------------------------------------------------------------------------- |
## | state (bh, r, dkc) | 4·2⁻²⁴·a + abs(k)·(β·2·Dk·2⁻²⁴·kvAbs + 2·2⁻²⁴·β·(abs(v)+abs(kv)) + 2·2⁻²⁴·abs(δ)) + 4·2⁻²⁴·(a + b) |
## | y (bh, r)          | 2·u_fam·abs(y) + (2·Dk·2⁻²⁴ + 2·2⁻²¹)·yAbs + 2·2⁻²⁴·abs(y) + 2⁻²⁵                                  |
##
## | term             | covers                                                     |
## | ---------------- | ---------------------------------------------------------- |
## | 2·u_fam·abs(y)   | both sides round the same fp32 value once                  |
## | (2·Dk·2⁻²⁴)·yAbs | the two dot orders                                         |
## | 2·2⁻²¹·yAbs      | the rsqrt-vs-divide q̃ difference                          |
## | 2⁻²⁵             | the fp16 subnormal grid floor, also covering the bf16 grid |
##
## - The measured divergence justifies the model, never sets the bar
## - Adjudication recorded on Apple M4 Max, fresh seeded inputs
##
## | claim            | measured                                                                                                                                                                                     |
## | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | reproduction     | the earlier full-mixer comparison's failure does not reproduce outside its frozen recording, the same dispatcher on fresh inputs is bit-exact (qkv/conv/l2norm/beta), g within 1.19 fp32 ulp |
## | family dtype     | fp16, both spellings inside the stated bars, worst usage 0.584 of the state bar and 0.487 of the y bar                                                                                       |
## | band consequence | fp16's unit roundoff 2⁻¹¹ vs bf16's 2⁻⁸ makes the y band ~8x tighter at equal usage, state arithmetic stays fp32, the family operands stay in fp16 normal range                              |
##
## - 40 seeded random cases per (family dtype, Dk) combination, Dk ∈ {32, 64}, over Hv = 2 value heads (hkRatio 2), Dv = 16, TileR = 8
## - Family dtypes bf16 and fp16, both inside their stated bar
## - One case per combination repeats with identical inputs and must stay bit-identical across launches
##
## - Every launch is followed by sentinel checks, kernel-written buffers inside their extents, kernel-read buffers bit-identical
## - Total device work is 4 combinations × 42 launches (40 + 2 determinism relaunches) of 4 threadgroups × 32 lanes
##
## Run (from the op worktree root):
##   nim c -r -d:release --hints:off --warnings:off -o:build/poc_gdn_step workspace/positron/tests/poc_ceramic_gdn_step.nim

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import naive/naive_rng
import naive/naive_tensors
import naive/naive_gdn
import naive/naive_metrics

# ─── Device code, the per-tile decode step, bf16 and fp16 spellings ───

proc gdnWidenBf16[A, B: static MmaAtom; R, C: static int](
    dst: var RtLeft[float32, R, C, A],
    src: RtLeft[bfloat16, R, C, B]) {.device.} =
  ## Exact bf16 → fp32 widening, walking each tile's own atom lane→element
  ## mapping. The 8×8×8 Apple atoms share one lane→element geometry, so
  ## the fragment indices agree elementwise.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst.frags[n][m].frag[v] = src.frags[n][m].frag[v].float32

proc gdnWidenF16[A, B: static MmaAtom; R, C: static int](
    dst: var RtLeft[float32, R, C, A],
    src: RtLeft[float16, R, C, B]) {.device.} =
  ## Exact fp16 → fp32 widening, same lane→element walk as `gdnWidenBf16`.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst.frags[n][m].frag[v] = src.frags[n][m].frag[v].float32

proc gdnStepCoreBf16(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[bfloat16],      # (B·Hv, Dv) bf16 core output
    k: ptr UncheckedArray[bfloat16],      # (B·Hk, Dk) bf16, one key head
    q: ptr UncheckedArray[bfloat16],      # (B·Hk, Dk) bf16, one key head
    v: ptr UncheckedArray[bfloat16],      # (B·Hv, Dv) bf16, one per head
    g: ptr UncheckedArray[float32],       # (B·Hv,) f32 log-decay, one per head
    beta: ptr UncheckedArray[bfloat16],   # (B·Hv,) bf16, one per head
    Hv, Hk, hkRatio: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR: static int) {.device.} =
  ## One (bh, TileR-row) state tile of the gated delta-rule decode step.
  ##
  ##   S ← S·exp2(g·log2e) + k ⊗ (β·(v − (S·exp2(g·log2e))·k))    y ← S'·(q·Dk^-0.5)
  ##
  ## Contract:
  ## - all state arithmetic is fp32, the state never rounds
  ## - the decay applies before the kv read (the chain's step order)
  ## - the decay is the exp2 form (Metal has no exp device builtin), q̃
  ##   folds Dk^-0.5 in fp32 as the rsqrt-multiply form
  ##
  ## - q and k enter post-l2norm as bf16, v and beta are bf16, g is fp32
  ## - y rounds once to bf16, one lane per state row, written by the lanes
  ##   whose fragment column is 0
  ## - the state stores in place, fp32, no rounding
  ##
  ## - `bh` is the (sequence, value head) row block, `dvBlock` the Dv/TileR
  ##   row block, the grid-driven entries pass the threadgroup coordinates
  ## - generic only over the static shape, the engine's monomorphization
  ##   key erases static bindings, every (Dk, Dv, TileR) binding needs
  ##   its own call-site line
  let hk = ((bh mod Hv) div hkRatio) + ((bh div Hv) * Hk)
  let headLin = bh * Dv * Dk
  let yLin = bh * Dv
  let kLin = hk * Dk

  let glState = state.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dk, 1))
  let glK = k.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glQ = q.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glV = v.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1, 0))

  var s: rt_l(float32, TileR, Dk)
  var kT: rt_l(bfloat16, TileR, Dk)
  var qT: rt_l(bfloat16, TileR, Dk)
  var vT: rt_l(bfloat16, TileR, 8)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))
  kT.loadTile(glK, (kLin, 0, 0, 0))
  qT.loadTile(glQ, (kLin, 0, 0, 0))
  vT.loadTile(glV, (yLin, 0, dvBlock, 0))

  let dec = exp2(g[bh] * 1.4426950408889634'f32)
  s.mul(s, dec)

  # kv_mem[row] = Σ_dk S[row][dk]·k[dk] over the decayed state, the k tile
  # broadcasts one key vector over the tile rows, one row sum per lane
  var k32: rt_l(float32, TileR, Dk)
  k32.gdnWidenBf16(kT)
  var prod: rt_l(float32, TileR, Dk)
  prod.mul(s, k32)
  var kvVec: rv(float32, TileR, Dk)
  kvVec.row_sum(prod)
  let kvMem = kvVec.data[0]

  let v32 = vT.frags[0][0].frag[0].float32
  let delta = beta[bh].float32 * (v32 - kvMem)

  const atom = getTileConfig(float32, float32)
  static:
    doAssert TileR == 8, "the y store maps one atom row block per column block"
    doAssert Dv mod TileR == 0, "the column grid covers Dv in whole row blocks"
    doAssert TileR mod atom.getM() == 0 and Dk mod atom.getN() == 0
  const rowTiles = TileR div atom.getM()
  const colTiles = Dk div atom.getN()
  const vpt = atom.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        s.frags[n][m].frag[v] =
          s.frags[n][m].frag[v] + k32.frags[n][m].frag[v] * delta

  let scale = rsqrt(float32(Dk))
  var q32: rt_l(float32, TileR, Dk)
  q32.gdnWidenBf16(qT)
  var oProd: rt_l(float32, TileR, Dk)
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        oProd.frags[n][m].frag[v] =
          s.frags[n][m].frag[v] * (q32.frags[n][m].frag[v] * scale)
  var oVec: rv(float32, TileR, Dk)
  oVec.row_sum(oProd)
  let oVal = oVec.data[0]

  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (lane, 0)).toIntVal()
  let rowIn = cell mod 8
  let colIn = cell div 8
  if colIn == 0:
    y[yLin + dvBlock * 8 + int32(rowIn)] = oVal.bfloat16
  glState.storeTile(s, (headLin, 0, dvBlock, 0))

proc gdnStepCoreF16(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[float16],       # (B·Hv, Dv) fp16 core output
    k: ptr UncheckedArray[float16],       # (B·Hk, Dk) fp16, one key head
    q: ptr UncheckedArray[float16],       # (B·Hk, Dk) fp16, one key head
    v: ptr UncheckedArray[float16],       # (B·Hv, Dv) fp16, one per head
    g: ptr UncheckedArray[float32],       # (B·Hv,) f32 log-decay, one per head
    beta: ptr UncheckedArray[float16],    # (B·Hv,) fp16, one per head
    Hv, Hk, hkRatio: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR: static int) {.device.} =
  ## `gdnStepCoreBf16` with the fp16 family dtype, the same fp32 state arithmetic,
  ## fp16 loads and one fp16 y rounding. The 8×8×8 fp16 atom shares the bf16
  ## atom's lane→element geometry, the tile walk is shared
  let hk = ((bh mod Hv) div hkRatio) + ((bh div Hv) * Hk)
  let headLin = bh * Dv * Dk
  let yLin = bh * Dv
  let kLin = hk * Dk

  let glState = state.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dk, 1))
  let glK = k.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glQ = q.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glV = v.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1, 0))

  var s: rt_l(float32, TileR, Dk)
  var kT: rt_l(float16, TileR, Dk)
  var qT: rt_l(float16, TileR, Dk)
  var vT: rt_l(float16, TileR, 8)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))
  kT.loadTile(glK, (kLin, 0, 0, 0))
  qT.loadTile(glQ, (kLin, 0, 0, 0))
  vT.loadTile(glV, (yLin, 0, dvBlock, 0))

  let dec = exp2(g[bh] * 1.4426950408889634'f32)
  s.mul(s, dec)

  var k32: rt_l(float32, TileR, Dk)
  k32.gdnWidenF16(kT)
  var prod: rt_l(float32, TileR, Dk)
  prod.mul(s, k32)
  var kvVec: rv(float32, TileR, Dk)
  kvVec.row_sum(prod)
  let kvMem = kvVec.data[0]

  let v32 = vT.frags[0][0].frag[0].float32
  let delta = beta[bh].float32 * (v32 - kvMem)

  const atom = getTileConfig(float32, float32)
  static:
    doAssert TileR == 8, "the y store maps one atom row block per column block"
    doAssert Dv mod TileR == 0, "the column grid covers Dv in whole row blocks"
    doAssert TileR mod atom.getM() == 0 and Dk mod atom.getN() == 0
  const rowTiles = TileR div atom.getM()
  const colTiles = Dk div atom.getN()
  const vpt = atom.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        s.frags[n][m].frag[v] =
          s.frags[n][m].frag[v] + k32.frags[n][m].frag[v] * delta

  let scale = rsqrt(float32(Dk))
  var q32: rt_l(float32, TileR, Dk)
  q32.gdnWidenF16(qT)
  var oProd: rt_l(float32, TileR, Dk)
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        oProd.frags[n][m].frag[v] =
          s.frags[n][m].frag[v] * (q32.frags[n][m].frag[v] * scale)
  var oVec: rv(float32, TileR, Dk)
  oVec.row_sum(oProd)
  let oVal = oVec.data[0]

  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (lane, 0)).toIntVal()
  let rowIn = cell mod 8
  let colIn = cell div 8
  if colIn == 0:
    y[yLin + dvBlock * 8 + int32(rowIn)] = oVal.float16
  glState.storeTile(s, (headLin, 0, dvBlock, 0))

const PocMsl = metal:
  proc poc_gdn_step_bf16_dk32(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[bfloat16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio: int32) {.global.} =
    let dvBlock = int32(threadgroup_position_in_grid.x)
    let bh = int32(threadgroup_position_in_grid.y)
    gdnStepCoreBf16(state, y, k, q, v, g, beta, Hv, Hk, hkRatio,
      dvBlock, bh, 32, 16, 8)

  proc poc_gdn_step_bf16_dk64(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[bfloat16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio: int32) {.global.} =
    let dvBlock = int32(threadgroup_position_in_grid.x)
    let bh = int32(threadgroup_position_in_grid.y)
    gdnStepCoreBf16(state, y, k, q, v, g, beta, Hv, Hk, hkRatio,
      dvBlock, bh, 64, 16, 8)

  proc poc_gdn_step_fp16_dk32(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[float16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio: int32) {.global.} =
    let dvBlock = int32(threadgroup_position_in_grid.x)
    let bh = int32(threadgroup_position_in_grid.y)
    gdnStepCoreF16(state, y, k, q, v, g, beta, Hv, Hk, hkRatio,
      dvBlock, bh, 32, 16, 8)

  proc poc_gdn_step_fp16_dk64(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[float16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio: int32) {.global.} =
    let dvBlock = int32(threadgroup_position_in_grid.x)
    let bh = int32(threadgroup_position_in_grid.y)
    gdnStepCoreF16(state, y, k, q, v, g, beta, Hv, Hk, hkRatio,
      dvBlock, bh, 64, 16, 8)

# ─── Host, page-padded device buffers, the tolerance model, the comparison ───

const HostPageSize = 16384  # Metal no-copy binding alignment

type PageBuf[T] = object
  data: pointer
  elems: int

proc posixMemalign(memptr: ptr pointer; alignment, size: csize_t): cint
  {.importc: "posix_memalign", header: "<stdlib.h>".}

proc freeShared(p: pointer) {.importc: "free", header: "<stdlib.h>".}

proc allocPageBuf[T](elems: int): PageBuf[T] =
  ## Stored element count rounded up so the byte extent is a host-page multiple,
  ## (the engine's no-copy binding contract), bytes past the extent stay untouched
  ## scratch by the kernel
  let nbytes = elems * sizeof(T)
  let rounded = (nbytes + HostPageSize - 1) div HostPageSize * HostPageSize
  var p: pointer = nil
  doAssert posixMemalign(addr p, csize_t(HostPageSize), csize_t(rounded)) == 0
  zeroMem(p, rounded)
  PageBuf[T](data: p, elems: rounded div sizeof(T))

func hostPtr[T](buf: PageBuf[T]): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](buf.data)

func pa[T](buf: PageBuf[T]): PtrArg[T] =
  PtrArg[T](buf: buf.hostPtr, len: buf.elems, off: 0)

proc freePageBuf[T](buf: PageBuf[T]) =
  freeShared(buf.data)

type Family = enum
  famBf16, famF16

proc toFamBits(fam: Family, x: float32): uint16 =
  ## Returns the family-dtype round-to-nearest-even bit pattern of an fp32 value.
  if fam == famBf16: f32ToBf16(x) else: fp32ToFp16(x)

proc famWiden(fam: Family, h: uint16): float32 =
  ## Returns the exact fp32 widening of a family dtype bit pattern.
  if fam == famBf16: bf16ToF32(h) else: fp16ToFp32(h)

proc famName(fam: Family): string =
  if fam == famBf16: "bf16" else: "fp16"

const
  U32 = 5.9604644775390625e-8        # 2⁻²⁴, the fp32 unit roundoff
  RelDecay = 4.0 * U32               # exp vs exp2(g·log2e) relative bound
  RelQScale = 2.0 * 4.76837158203125e-7  # 2·2⁻²¹, rsqrt vs divide, relative
  UBf16 = 3.90625e-3                 # 2⁻⁸, the bf16 unit roundoff
  UF16 = 4.8828125e-4                # 2⁻¹¹, the fp16 unit roundoff
  FloorSub = 2.9802322387695312e-8   # 2⁻²⁵, half the constant fp16 subnormal ulp,
                                     # the rounding floor once |y| falls subnormal

proc famUlp(fam: Family, v: float64): float64 =
  ## Width of one family-dtype ulp at a nonzero normal |v|.
  if v == 0.0: return 0.0
  let (mant, exp10) = frexp(abs(v))
  doAssert mant >= 0.5 and mant < 1.0
  let floorExp = exp10 - 1           # floor(log2|v|), the binary exponent of |v|
  let mantBits = if fam == famBf16: 7 else: 10
  result = pow(2.0, float64(floorExp - mantBits))

proc checkBitsUnchanged(buf: PageBuf[uint16], bits: seq[uint16]) =
  ## A kernel-read buffer must stay bit-identical, the host memory
  ## is the device memory under the no-copy binding
  let view = cast[ptr UncheckedArray[uint16]](buf.data)
  for i in 0 ..< bits.len:
    doAssert view[i] == bits[i], "kernel-read buffer modified at element " & $i

proc checkF32Unchanged(buf: PageBuf[float32], vals: seq[float32]) =
  ## A kernel-read fp32 buffer must stay bit-identical.
  for i in 0 ..< vals.len:
    doAssert buf.hostPtr[i] == vals[i], "kernel-read buffer modified at element " & $i

proc runCombo(engine: HwEngine, fam: Family, dk: int, cases: int, seed: uint64) =
  ## One (family dtype, Dk) combination judged per element against the naive
  ## reference bars, the case-0 inputs relaunched bit-identical
  const Hv = 2
  const Hk = 1
  const HkRatio = 2
  const Dv = 16
  const TileR = 8
  let stateElems = Hv * Dv * dk
  let kernelName = "poc_gdn_step_" & famName(fam) & (if dk == 32: "_dk32" else: "_dk64")

  var stateB = allocPageBuf[float32](stateElems)
  var yB = allocPageBuf[uint16](Hv * Dv)
  var kB = allocPageBuf[uint16](Hk * dk)
  var qB = allocPageBuf[uint16](Hk * dk)
  var vB = allocPageBuf[uint16](Hv * Dv)
  var gB = allocPageBuf[float32](Hv)
  var betaB = allocPageBuf[uint16](Hv)
  defer:
    freePageBuf(stateB); freePageBuf(yB); freePageBuf(kB); freePageBuf(qB)
    freePageBuf(vB); freePageBuf(gB); freePageBuf(betaB)
  # The run sugar's output argument binds a var, the pointer args hoist
  # to combo-scope vars and feed every launch of this combination
  var statePA = stateB.pa()
  var yPA = yB.pa()
  var kPA = kB.pa()
  var qPA = qB.pa()
  var vPA = vB.pa()
  var gPA = gB.pa()
  var betaPA = betaB.pa()

  var worstState = 0.0'f64
  var worstStateUse = 0.0'f64
  var worstYUse = 0.0'f64
  var worstYUlp = 0.0'f64
  var yExact = 0
  var yTotal = 0

  var rng = initNaiveRng(seed)
  for caseId in 0 ..< cases:
    # ── seeded inputs ──
    let nQK = Hk * dk
    var qBits = newSeq[uint16](nQK)
    var kBits = newSeq[uint16](nQK)
    var vBits = newSeq[uint16](Hv * Dv)
    var betaBits = newSeq[uint16](Hv)
    var gVals = newSeq[float32](Hv)
    for i in 0 ..< nQK:
      qBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
      kBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
    for i in 0 ..< Hv * Dv:
      vBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
    for h in 0 ..< Hv:
      betaBits[h] = toFamBits(fam, rng.nextF32(0.2'f32, 0.8'f32))
      gVals[h] = rng.nextF32(-3.0'f32, -0.1'f32)
    var stateVals = newSeq[float32](stateElems)
    for i in 0 ..< stateElems:
      stateVals[i] = rng.nextF32(-1.0'f32, 1.0'f32)

    # ── kernel step ──
    copyMem(stateB.data, unsafeAddr stateVals[0], stateElems * 4)
    copyMem(kB.data, unsafeAddr kBits[0], nQK * 2)
    copyMem(qB.data, unsafeAddr qBits[0], nQK * 2)
    copyMem(vB.data, unsafeAddr vBits[0], Hv * Dv * 2)
    copyMem(gB.data, unsafeAddr gVals[0], Hv * 4)
    copyMem(betaB.data, unsafeAddr betaBits[0], Hv * 2)
    engine.run << (grid: (Dv div TileR, Hv, 1), blk: (32, 1, 1)) >>
      (kernelName, statePA,
        (yPA, kPA, qPA, vPA, betaPA, gPA,
         int32(Hv), int32(Hk), int32(HkRatio)))

    # sentinels, kernel-written buffers stay in extent, kernel-read
    # buffers stay bit-identical
    for i in 0 ..< yB.elems:
      if i >= Hv * Dv:
        doAssert yB.hostPtr[i] == 0, "y written past its extent"
    for i in stateElems ..< stateB.elems:
      doAssert stateB.hostPtr[i] == 0.0'f32, "state written past its extent"
    checkBitsUnchanged(kB, kBits)
    checkBitsUnchanged(qB, qBits)
    checkBitsUnchanged(vB, vBits)
    checkF32Unchanged(gB, gVals)
    checkBitsUnchanged(betaB, betaBits)

    # ── naive reference, same fp32 width, same widened inputs ──
    var qF = newSeq[float32](nQK)
    var kF = newSeq[float32](nQK)
    var vF = newSeq[float32](Hv * Dv)
    var betaF = newSeq[float32](Hv)
    for i in 0 ..< nQK:
      qF[i] = famWiden(fam, qBits[i])
      kF[i] = famWiden(fam, kBits[i])
    for i in 0 ..< Hv * Dv:
      vF[i] = famWiden(fam, vBits[i])
    for h in 0 ..< Hv:
      betaF[h] = famWiden(fam, betaBits[h])
    let stateStart = stateVals
    var stateNaive = NaiveCube[float32](planes: Hv, rows: Dv, cols: dk)
    stateNaive.data = stateVals
    var yNaive = NaiveMat[float32](rows: Hv, cols: Dv)
    yNaive.data = newSeq[float32](Hv * Dv)
    var qMat = NaiveMat[float32](rows: Hk, cols: dk)
    qMat.data = qF
    var kMat = NaiveMat[float32](rows: Hk, cols: dk)
    kMat.data = kF
    var vMat = NaiveMat[float32](rows: Hv, cols: Dv)
    vMat.data = vF
    gdnDecodeStep(stateNaive, yNaive, qMat, kMat, vMat, betaF, gVals,
      Hv, Hk, HkRatio)

    # ── per-row tolerance-model terms on the naive side ──
    var deltaN = newSeq[float64](Dv)
    var dDelta = newSeq[float64](Dv)
    for bh in 0 ..< Hv:
      let hk = (bh mod Hv) div HkRatio + (bh div Hv) * Hk
      let gamma = exp(gVals[bh].float64)
      for r in 0 ..< Dv:
        var kvN = 0.0'f64
        var kvAbs = 0.0'f64
        for c in 0 ..< dk:
          let t = gamma * stateStart[bh * Dv * dk + r * dk + c].float64 *
            kF[hk * dk + c].float64
          kvN += t
          kvAbs += abs(t)
        let d = betaF[bh].float64 * (vF[bh * Dv + r].float64 - kvN)
        deltaN[r] = d
        dDelta[r] = betaF[bh].float64 * (2.0 * dk.float64 * U32 * kvAbs) +
          2.0 * U32 * betaF[bh].float64 *
            (abs(vF[bh * Dv + r].float64) + abs(kvN)) + 2.0 * U32 * abs(d)

        # state bar, element (bh, r, dkc)
        for c in 0 ..< dk:
          let a = abs(gamma * stateStart[bh * Dv * dk + r * dk + c].float64)
          let b = abs(kF[hk * dk + c].float64 * d)
          let barS = RelDecay * a + abs(kF[hk * dk + c].float64) * dDelta[r] +
            4.0 * U32 * (a + b)
          let got = stateB.hostPtr[bh * Dv * dk + r * dk + c].float64
          let want = stateNaive.data[bh * Dv * dk + r * dk + c].float64
          let diff = abs(got - want)
          doAssert diff <= barS,
            &"state outside the bar at (bh {bh}, r {r}, dkc {c}): " &
            &"{diff:.3e} > {barS:.3e}"
          worstState = max(worstState, diff)
          if barS > 0.0: worstStateUse = max(worstStateUse, diff / barS)

        # y bar, element (bh, r)
        var yAbs = 0.0'f64
        for c in 0 ..< dk:
          let qs = qF[hk * dk + c].float64 / sqrt(dk.float32).float64
          yAbs += abs(stateNaive.data[bh * Dv * dk + r * dk + c].float64 * qs)
        let yN = yNaive.data[bh * Dv + r].float64
        let uFam = if fam == famBf16: UBf16 else: UF16
        let barY = 2.0 * uFam * abs(yN) +
          (2.0 * dk.float64 * U32 + RelQScale) * yAbs +
          2.0 * U32 * abs(yN) + FloorSub
        let yGot = famWiden(fam, yB.hostPtr[bh * Dv + r]).float64
        let diff = abs(yGot - yN)
        doAssert diff <= barY,
          &"y outside the bar at (bh {bh}, r {r}): {diff:.3e} > {barY:.3e}"
        worstYUse = max(worstYUse, diff / barY)
        let uAt = famUlp(fam, yN)
        if uAt > 0.0 and diff > 0.0:
          worstYUlp = max(worstYUlp, diff / uAt)
        if diff == 0.0:
          inc yExact
        inc yTotal

  # ── case-0 relaunch: identical inputs must be bit-identical ──
  block determinism:
    var rng0 = initNaiveRng(seed)
    let nQK = Hk * dk
    var qBits = newSeq[uint16](nQK)
    var kBits = newSeq[uint16](nQK)
    var vBits = newSeq[uint16](Hv * Dv)
    var betaBits = newSeq[uint16](Hv)
    var gVals = newSeq[float32](Hv)
    for i in 0 ..< nQK:
      qBits[i] = toFamBits(fam, rng0.nextF32(-1.0'f32, 1.0'f32))
      kBits[i] = toFamBits(fam, rng0.nextF32(-1.0'f32, 1.0'f32))
    for i in 0 ..< Hv * Dv:
      vBits[i] = toFamBits(fam, rng0.nextF32(-1.0'f32, 1.0'f32))
    for h in 0 ..< Hv:
      betaBits[h] = toFamBits(fam, rng0.nextF32(0.2'f32, 0.8'f32))
      gVals[h] = rng0.nextF32(-3.0'f32, -0.1'f32)
    var stateVals = newSeq[float32](stateElems)
    for i in 0 ..< stateElems:
      stateVals[i] = rng0.nextF32(-1.0'f32, 1.0'f32)
    var stateSnap = newSeq[float32](stateElems)
    var ySnap = newSeq[uint16](Hv * Dv)
    # first pass of the relaunch pair
    copyMem(stateB.data, unsafeAddr stateVals[0], stateElems * 4)
    copyMem(kB.data, unsafeAddr kBits[0], nQK * 2)
    copyMem(qB.data, unsafeAddr qBits[0], nQK * 2)
    copyMem(vB.data, unsafeAddr vBits[0], Hv * Dv * 2)
    copyMem(gB.data, unsafeAddr gVals[0], Hv * 4)
    copyMem(betaB.data, unsafeAddr betaBits[0], Hv * 2)
    engine.run << (grid: (Dv div TileR, Hv, 1), blk: (32, 1, 1)) >>
      (kernelName, statePA,
        (yPA, kPA, qPA, vPA, betaPA, gPA,
         int32(Hv), int32(Hk), int32(HkRatio)))
    copyMem(addr stateSnap[0], stateB.data, stateElems * 4)
    copyMem(addr ySnap[0], yB.data, Hv * Dv * 2)
    # second pass, same inputs
    copyMem(stateB.data, unsafeAddr stateVals[0], stateElems * 4)
    copyMem(kB.data, unsafeAddr kBits[0], nQK * 2)
    copyMem(qB.data, unsafeAddr qBits[0], nQK * 2)
    copyMem(vB.data, unsafeAddr vBits[0], Hv * Dv * 2)
    copyMem(gB.data, unsafeAddr gVals[0], Hv * 4)
    copyMem(betaB.data, unsafeAddr betaBits[0], Hv * 2)
    engine.run << (grid: (Dv div TileR, Hv, 1), blk: (32, 1, 1)) >>
      (kernelName, statePA,
        (yPA, kPA, qPA, vPA, betaPA, gPA,
         int32(Hv), int32(Hk), int32(HkRatio)))
    for i in 0 ..< stateElems:
      doAssert stateB.hostPtr[i] == stateSnap[i], "state differs run to run"
    for i in 0 ..< Hv * Dv:
      doAssert yB.hostPtr[i] == ySnap[i], "y differs run to run"

  echo &"[{famName(fam)} Dk={dk}] state worst |ΔS| {worstState:.3e}, " &
    &"worst bar usage {worstStateUse:.3f} | y worst {worstYUlp:.2f} " &
    &"{famName(fam)} ulp, bit-exact {yExact}/{yTotal}, worst bar usage {worstYUse:.3f}"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(PocMsl)
  const Cases = 40
  let t0 = epochTime()
  runCombo(engine, famBf16, 32, Cases, 0xC04D0401'u64)
  runCombo(engine, famBf16, 64, Cases, 0xC04D0402'u64)
  runCombo(engine, famF16, 32, Cases, 0xC04D0403'u64)
  runCombo(engine, famF16, 64, Cases, 0xC04D0404'u64)
  let secs = epochTime() - t0
  echo &"wall clock: {secs:.2f} s for {4 * (Cases + 2)} launches"
  echo "POC VERDICT: all combinations inside the stated per-element bars"

main()
