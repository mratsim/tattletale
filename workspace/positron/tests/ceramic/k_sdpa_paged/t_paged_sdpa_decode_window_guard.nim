# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Windowed-decode launch guard: `paged_sdpa_check_window` must raise on a windowed decode whose windowSize is not positive before any launch.
##
## - non-positive width: the attention band empties, no key survives,
##   the softmax sum never accumulates and the output would store NaN
## - valid width (windowSize = 16): the decode replays against a float64
##   host reference over the attended entries
##
## Run from the repository root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/t_paged_sdpa_decode_window_guard --nimcache:nimcache/t_paged_sdpa_decode_window_guard \
##     workspace/positron/tests/ceramic/k_sdpa_paged/t_paged_sdpa_decode_window_guard.nim

import std/[math, random]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils
import workspace/positron/src/kernels/k_tile_sdpa_paged
import workspace/positron/tests/ceramic/k_sdpa_paged/kernel_sdpa_paged_template

const
  D = 16             # head dim, f32 path like the base suite's tiny_f32
  PageSize = 32      # one kv tile per page, matching the fixture suite's schedule.
  Seqlen = 40        # spans two pages, a full tile then an 8-entry tail.
  Window = 16        # valid width, entries [Seqlen - Window, Seqlen)

proc makeFlatI32Tensor(n: int): auto =
  ## Rank-1 int32 host tensor for cache seqlens. Built by hand:
  ## makeHostTensor rejects rank-1 shapes (see sdpa_paged_fixtures.nim).
  let layout = make_layout((n,))
  HostTensor[int32, typeof(layout.shape), typeof(layout.stride)](
    data: cast[ptr UncheckedArray[int32]](allocShared(n * sizeof(int32))),
    offset: 0,
    layout: layout)

proc fillPagedPool[T, Sh, St](ht: HostTensor[T, Sh, St], rng: var Rand) =
  ## Fill a (numPages, numLayers, pageSize, Hkv, D) pool with deterministic
  ## values in [1, 2] on the logical entries.
  for p in 0 ..< ht.layout.shape[0]:
    for l in 0 ..< ht.layout.shape[1]:
      for r in 0 ..< ht.layout.shape[2]:
        for h in 0 ..< ht.layout.shape[3]:
          for j in 0 ..< D:
            ht[p, l, r, h, j] = rng.rand(1.0 .. 2.0)

proc windowedReference[Tq, Sq, Stq, Tk, Sk, Stk, Tv, Sv, Stv](
    qh: HostTensor[Tq, Sq, Stq], kh: HostTensor[Tk, Sk, Stk], vh: HostTensor[Tv, Sv, Stv],
    windowSize: int): seq[float64] =
  ## float64 host reference: softmax(scale * q @ K^T) @ V over the gathered
  ## entries [Seqlen - windowSize, Seqlen), entry t at pool page
  ## bt[t div PageSize] entry t mod PageSize (page ids 1 and 2, layer 0).
  let scale = 0.125'f64
  var scores: seq[float64] = @[]
  for t in (Seqlen - windowSize) ..< Seqlen:
    let
      page = if t < PageSize: 1 else: 2
      r = t mod PageSize
    var s = 0.0
    for j in 0 ..< D:
      s += scale * qh[0, 0, 0, j].float64 * kh[page, 0, r, 0, j].float64
    scores.add s
  let m = max(scores)
  var denom = 0.0
  for s in scores:
    denom += exp(s - m)
  for j in 0 ..< D:
    var acc = 0.0
    for i in 0 ..< windowSize:
      let
        t = Seqlen - windowSize + i
        page = if t < PageSize: 1 else: 2
        r = t mod PageSize
      acc += exp(scores[i] - m) / denom * vh[page, 0, r, 0, j].float64
    result.add acc

proc replayWindowGuard() =
  var rng = initRand(11)
  var engine = newEngine(runMetal)

  let
    kh = makeHostTensor(float32, (3, 1, PageSize, 1, D), LayoutRight)
    vh = makeHostTensor(float32, (3, 1, PageSize, 1, D), LayoutRight)
    qh = makeHostTensor(float32, (1, 1, 1, D), LayoutRight)
    oh = makeHostTensor(float32, (1, 1, 1, D), LayoutRight)
  fillPagedPool(kh, rng)
  fillPagedPool(vh, rng)
  for h in 0 ..< D:
    qh[0, 0, 0, h] = rng.rand(1.0 .. 2.0)

  let
    bt = makeHostTensor(int32, (1, 2), LayoutRight)
  var cs = makeFlatI32Tensor(1)
  bt[0, 0] = 1
  bt[0, 1] = 2   # page ids 1 and 2, pool pages 0 and 2 stay unread
  cs[0] = Seqlen

  proc launchWindowed(windowSize: int): auto =
    let
      qDev = engine.bridge(qh)
      kDev = engine.bridge(kh)
      vDev = engine.bridge(vh)
      btDev = engine.bridge(bt)
      csDev = engine.bridge(cs)
      oDev = engine.bridge(oh)
    let
      scaleVal = 0.125'f32
      layerVal = int32(0)
      win = windowSize
    engine.launchDeviceFn sdpaPagedCombo(
        oDev, qDev, kDev, vDev, btDev, csDev, scaleVal, D, D,
        PageSize, true, win, layerVal) do:
      grid = (1, 1, 1)
      blk = (32, 1)
    var outHost = makeHostTensor(float32, (1, 1, 1, D), LayoutRight)
    engine.copyToHostFromDevice(addr outHost.data[0], oDev.data, outHost.len())
    outHost

  # a valid width replays against the float64 host reference
  block:
    let got = launchWindowed(Window)
    let want = windowedReference(qh, kh, vh, Window)
    for j in 0 ..< D:
      let
        g = got[0, 0, 0, j].float64
        w = want[j]
      doAssert abs(g - w) <= 1e-4 + 1e-3 * abs(w),
        "windowed decode mismatch at " & $j & ": got " & $g & " want " & $w
    echo "windowed replay ok: windowSize = ", Window, " matches the host reference"

  # a non-positive width raises before any launch
  block:
    var refused = 0
    for bad in [0, -4]:
      try:
        paged_sdpa_check_window(windowed = true, windowSize = bad)
        doAssert false, "windowSize = " & $bad & " must raise before launch"
      except AssertionDefect:
        refused.inc
    doAssert refused == 2

  # a valid width and the full-context sentinel pass the guard
  paged_sdpa_check_window(windowed = true, windowSize = Window)
  paged_sdpa_check_window(windowed = false, windowSize = low(int32))

replayWindowGuard()
