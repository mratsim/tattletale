{.experimental: "callOperator".}
## Benchmark: all iteration methods for nested-layout copy on CPU
## Layout setup matches gemm_cute_v20.nim production.

import std/[monotimes, times, math, random, strutils, stats, algorithm, typetraits]
import ../../src/int_tuples, ../../src/layouts, ../../src/layout_algebra, ../../src/tensors
export int_tuples, layouts, layout_algebra, tensors
import ../../src/macros/static_for

import ./laser01_global, ./laser02_pertensor, ./laser03_nested_forloops, ./laser05_fusedpertensor, ./gemm_packing_loop_explicit
import ./kernel_copy
import ./transpose2d

func toArray*(t: tuple): auto =
  ## Convert a tuple to array[tupleLen, int].
  const N = tupleLen(typeof(t))
  var a: array[N, int]
  staticFor i, 0, N:
    a[i] = int(t[i])
  a

proc xorHash*(data: openArray[float32]): uint32 =
  result = 0
  for i in 0 ..< data.len:
    result = result xor cast[uint32](data[i])

template copy_flatIndex(dst, src: typed) =
  let N = size(dst)
  for i in 0 ..< N:
    dst(i) = src(i)

template bench*(label: string; nSamples, opCount: int; body: untyped) =
  block:
    var st: RunningStat
    for _ in 0 ..< nSamples:
      let t0 = getMonoTime()
      body
      let t1 = getMonoTime()
      st.push(float64((t1 - t0).inNanoseconds) * 1e-9)
    let mean_s = st.mean()
    let gm_s = (float64(opCount) / 1e9) / mean_s
    echo label, ": ", (mean_s * 1e6).formatFloat(ffDecimal, 2), " us  ",
      gm_s.formatFloat(ffDecimal, 3), " GMEMOPS/s"

proc main() =
  const nb_samples = 100
  const nr = 16
  const kc = 512
  const npT = 64
  const mr = 6
  const mpT = 32
  const mc = mpT * mr
  const bElems = npT * kc * nr
  const aElems = mpT * kc * mr

  # ── Production-matched layouts ──
  let packALay = make_layout((mpT, kc, mr), LayoutRight)
  let packBLay = make_layout((npT, kc, nr), LayoutRight)
  var packA_buf = newSeq[float32](aElems)
  var packB_buf = newSeq[float32](bElems)
  var packA = make_view(packA_buf, packALay)
  var packB = make_view(packB_buf, packBLay)

  let panelA_lay = make_layout((mc, kc), (1, mc))
  let panelB_lay = make_layout((kc, npT*nr), (1, kc))

  let srcA_zd = zipped_divide(panelA_lay, (mr, 1))
  let dstA_zd = make_layout(((mr, 1), (mpT, kc)), ((1, mr), (mr * kc, mr)))
  let srcB_zd = zipped_divide(panelB_lay, (1, nr))
  let dstB_zd = make_layout(((1, nr), (kc, npT)), ((1, 1), (nr, nr * kc)))

  var panelA = newSeq[float32](int(cosize(panelA_lay)))
  var panelB = newSeq[float32](int(cosize(panelB_lay)))
  randomize(42)
  for i in 0 ..< panelA.len: panelA[i] = rand(1.0'f32)
  for i in 0 ..< panelB.len: panelB[i] = rand(1.0'f32)

  let panelB_v = make_view(panelB, panelB_lay)
  let srcB_tv = make_view(panelB_v, srcB_zd)
  var dstB_tv = make_view(packB, dstB_zd)

  let panelA_v = make_view(panelA, panelA_lay)
  let srcA_tv = make_view(panelA_v, srcA_zd)
  var dstA_tv = make_view(packA, dstA_zd)

  # Arrays extracted from real layouts (same for all functions)
  let bShape = toArray(flatten(srcB_tv.layout.shape))
  let bDstSt = toArray(flatten(dstB_tv.layout.stride))
  let bSrcSt = toArray(flatten(srcB_tv.layout.stride))
  let aShape = toArray(flatten(srcA_tv.layout.shape))
  let aDstSt = toArray(flatten(dstA_tv.layout.stride))
  let aSrcSt = toArray(flatten(srcA_tv.layout.stride))

  echo "=== Nested-layout copies (", nb_samples, " samples each) ==="

  # ── B-packing ──
  echo "\n-- B packing (", bElems, " floats) --"
  copy_flatIndex(dstB_tv, srcB_tv)
  let bHash_ref = xorHash(packB_buf)
  packB_buf.fill(0)
  copySameShape(dstB_tv, srcB_tv)
  doAssert xorHash(packB_buf) == bHash_ref
  packB_buf.fill(0)

  bench("autoresearch  ", nb_samples, bElems):
    copy_autoresearch_B(packB_buf, panelB, 1, kc, npT, kc, nr)
  doAssert xorHash(packB_buf) == bHash_ref, "autoresearch B"
  packB_buf.fill(0)

  bench("laser01 wheel  ", nb_samples, bElems):
    copy_laser01(packB_buf, bDstSt, panelB, bSrcSt, bShape)
  doAssert xorHash(packB_buf) == bHash_ref, "laser01 B"
  packB_buf.fill(0)

  bench("laser02 strided", nb_samples, bElems):
    copy_laser02(packB_buf, bDstSt, panelB, bSrcSt, bShape)
  doAssert xorHash(packB_buf) == bHash_ref, "laser02 B"
  packB_buf.fill(0)

  bench("laser03 nested for-loops  ", nb_samples, bElems):
    copy_laser03(packB_buf, bDstSt, panelB, bSrcSt, bShape)
  doAssert xorHash(packB_buf) == bHash_ref, "laser03 B"
  packB_buf.fill(0)

  bench("laser05 fused  ", nb_samples, bElems):
    copy_laser05(packB_buf, bDstSt, panelB, bSrcSt, bShape)
  doAssert xorHash(packB_buf) == bHash_ref, "laser05 B"
  packB_buf.fill(0)

  bench("crd2idx flat  ", nb_samples, bElems):
    copy_flatIndex(dstB_tv, srcB_tv)
  doAssert xorHash(packB_buf) == bHash_ref, "flat B"
  packB_buf.fill(0)

  bench("copySameShape", nb_samples, bElems):
    copySameShape(dstB_tv, srcB_tv)
  doAssert xorHash(packB_buf) == bHash_ref, "copySameShape B"
  packB_buf.fill(0)

  bench("copySameShape_08", nb_samples, bElems):
    copySameShape(dstB_tv, srcB_tv, 8)
  doAssert xorHash(packB_buf) == bHash_ref, "copySameShape_08 B"
  packB_buf.fill(0)

  bench("copySameShape_16", nb_samples, bElems):
    copySameShape(dstB_tv, srcB_tv, 16)
  doAssert xorHash(packB_buf) == bHash_ref, "copySameShape_16 B"
  packB_buf.fill(0)

  # ── A-packing ──
  echo "\n-- A packing (", aElems, " floats) --"
  copy_flatIndex(dstA_tv, srcA_tv)
  let aHash_ref = xorHash(packA_buf)
  packA_buf.fill(0)

  bench("autoresearch  ", nb_samples, aElems):
    copy_autoresearch_A(packA_buf, packA_buf, panelA, 1, mc, mpT, kc, mr, mc)
  doAssert xorHash(packA_buf) == aHash_ref, "autoresearch A"
  packA_buf.fill(0)

  bench("laser01 wheel  ", nb_samples, aElems):
    copy_laser01(packA_buf, aDstSt, panelA, aSrcSt, aShape)
  doAssert xorHash(packA_buf) == aHash_ref, "laser01 A"
  packA_buf.fill(0)

  bench("laser02 strided", nb_samples, aElems):
    copy_laser02(packA_buf, aDstSt, panelA, aSrcSt, aShape)
  doAssert xorHash(packA_buf) == aHash_ref, "laser02 A"
  packA_buf.fill(0)

  bench("laser03 nested for-loops  ", nb_samples, aElems):
    copy_laser03(packA_buf, aDstSt, panelA, aSrcSt, aShape)
  doAssert xorHash(packA_buf) == aHash_ref, "laser03 A"
  packA_buf.fill(0)

  bench("laser05 fused  ", nb_samples, aElems):
    copy_laser05(packA_buf, aDstSt, panelA, aSrcSt, aShape)
  doAssert xorHash(packA_buf) == aHash_ref, "laser05 A"
  packA_buf.fill(0)

  bench("crd2idx flat  ", nb_samples, aElems):
    copy_flatIndex(dstA_tv, srcA_tv)
  doAssert xorHash(packA_buf) == aHash_ref, "flat A"
  packA_buf.fill(0)

  bench("copySameShape", nb_samples, aElems):
    copySameShape(dstA_tv, srcA_tv)
  doAssert xorHash(packA_buf) == aHash_ref, "copySameShape A"
  packA_buf.fill(0)

  bench("copySameShape_08", nb_samples, aElems):
    copySameShape(dstA_tv, srcA_tv, 8)
  doAssert xorHash(packA_buf) == aHash_ref, "copySameShape_08 A"
  packA_buf.fill(0)

  bench("copySameShape_16", nb_samples, aElems):
    copySameShape(dstA_tv, srcA_tv, 16)
  doAssert xorHash(packA_buf) == aHash_ref, "copySameShape_16 A"
  packA_buf.fill(0)

  echo "\nAll hash checks passed."

  # ════════════════════════════════════════════════════════════════════
  #  Transpose
  # ════════════════════════════════════════════════════════════════════

  proc runTr(NR, NC: static int) =
    echo ""
    echo "-- Transpose ", NR, "×", NC, " (", NR*NC, " floats) --"
    let elems = NR * NC
    var src = newSeq[float32](elems)
    for i in 0 ..< elems:
      src[i] = rand(1.0'f32)

    let sv = make_view(src, make_layout((NR, NC), (1, NR)))

    var refH: uint32
    block:
      var dst = newSeq[float32](elems)
      let dv = make_view(dst, make_layout((NC, NR), (1, NC)))
      copy_flatIndex(dv, sv)
      refH = xorHash(dst)

    proc tx(label: string; refH: uint32) =
      block:
        var buf = newSeq[float32](elems)
        let dv = make_view(buf, make_layout((NC, NR), (1, NC)))
        bench(label, nb_samples, elems):
          discard
        doAssert xorHash(buf) == refH

    block:
      var buf = newSeq[float32](elems)
      let dv = make_view(buf, make_layout((NC, NR), (1, NC)))
      bench("crd2idx flat  ", nb_samples, elems):
        copy_flatIndex(dv, sv)
      doAssert xorHash(buf) == refH
    block:
      var buf = newSeq[float32](elems)
      bench("transpose_naive", nb_samples, elems):
        transpose2D_naive(addr buf[0], addr src[0], NR, NC)
      doAssert xorHash(buf) == refH
    block:
      var buf = newSeq[float32](elems)
      bench("transpose_laser", nb_samples, elems):
        transpose2D_laser(addr buf[0], addr src[0], NR, NC)
      doAssert xorHash(buf) == refH
    block:
      var buf = newSeq[float32](elems)
      bench("cache_block    ", nb_samples, elems):
        transpose2D_cacheBlock(addr buf[0], addr src[0], NR, NC)
      doAssert xorHash(buf) == refH
    block:
      var buf = newSeq[float32](elems)
      bench("cache_blk_prefetch ", nb_samples, elems):
        transpose2D_cacheBlockPrefetch(addr buf[0], addr src[0], NR, NC)
    block:
      var buf = newSeq[float32](elems)
      let dv = make_view(buf, make_layout((NC, NR), (1, NC)))
      bench("copyPermuted ", nb_samples, elems):
        copyPermuted(dv, sv, [1, 0])
      doAssert xorHash(buf) == refH
    block:
      var buf = newSeq[float32](elems)
      let dv = make_view(buf, make_layout((NC, NR), (1, NC)))
      bench("copyPermuted_16     ", nb_samples, elems):
        copyPermuted(dv, sv, [1, 0], 16)
      doAssert xorHash(buf) == refH

  runTr(4000, 2000)
  runTr(2000, 4000)
  echo "\nDone."

when isMainModule:
  main()
