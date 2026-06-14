{.experimental: "callOperator".}
## Benchmark: all iteration methods for nested-layout copy on CPU
## Layout setup matches gemm_cute_v20.nim production.

import std/[monotimes, times, math, random, strutils, stats, algorithm]
import ../src/int_tuples, ../src/layouts, ../src/layout_algebra, ../src/tensors

import ./bench_copy_cpu/laser01_global, ./bench_copy_cpu/laser02_pertensor, ./bench_copy_cpu/laser03_nested_forloops, ./bench_copy_cpu/laser05_fusedpertensor, ./bench_copy_cpu/gemm_packing_loop_explicit
import ../src/kernel_copy_cpu
import ./bench_copy_cpu/transpose2d
import ./bench_utils

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
    let gib_s = float64(opCount * 4) / (float64(1024)^3) / mean_s
    let ms = mean_s * 1e3
    echo label.alignLeft(26),
      ms.formatFloat(ffDecimal, 3).align(7), " ms  ",
      gm_s.formatFloat(ffDecimal, 3).align(7), " GMEMOPS/s  ",
      gib_s.formatFloat(ffDecimal, 2).align(7), " GiB/s"

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

  var panelA = newSeq[float32](cosize(panelA_lay).toIntVal())
  var panelB = newSeq[float32](cosize(panelB_lay).toIntVal())
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
  block:
    var buf = newSeq[float32](bElems)
    bench("copyMem (raw)", nb_samples, bElems):
      copyMem(addr buf[0], addr panelB[0], bElems * sizeof(float32))
  copy_flatIndex(dstB_tv, srcB_tv)
  let bHash_ref = xorHash(packB_buf)
  packB_buf.fill(0)
  copySameShape_cpu(dstB_tv, srcB_tv)
  doAssert xorHash(packB_buf) == bHash_ref
  packB_buf.fill(0)
  bench("autoresearch", nb_samples, bElems):
    copy_autoresearch_B(packB_buf, panelB, 1, kc, npT, kc, nr)
  doAssert xorHash(packB_buf) == bHash_ref, "autoresearch B"
  packB_buf.fill(0)

  bench("laser01 wheel", nb_samples, bElems):
    copy_laser01(packB_buf, bDstSt, panelB, bSrcSt, bShape)
  doAssert xorHash(packB_buf) == bHash_ref, "laser01 B"
  packB_buf.fill(0)

  bench("laser02 strided", nb_samples, bElems):
    copy_laser02(packB_buf, bDstSt, panelB, bSrcSt, bShape)
  doAssert xorHash(packB_buf) == bHash_ref, "laser02 B"
  packB_buf.fill(0)

  bench("laser03 nested for-loops", nb_samples, bElems):
    copy_laser03(packB_buf, bDstSt, panelB, bSrcSt, bShape)
  doAssert xorHash(packB_buf) == bHash_ref, "laser03 B"
  packB_buf.fill(0)

  bench("laser05 fused", nb_samples, bElems):
    copy_laser05(packB_buf, bDstSt, panelB, bSrcSt, bShape)
  doAssert xorHash(packB_buf) == bHash_ref, "laser05 B"
  packB_buf.fill(0)

  bench("crd2idx flat", nb_samples, bElems):
    copy_flatIndex(dstB_tv, srcB_tv)
  doAssert xorHash(packB_buf) == bHash_ref, "flat B"
  packB_buf.fill(0)

  bench("copySameShape_cpu", nb_samples, bElems):
    copySameShape_cpu(dstB_tv, srcB_tv)
  doAssert xorHash(packB_buf) == bHash_ref, "copySameShape_cpu B"
  packB_buf.fill(0)

  bench("copySameShape_08", nb_samples, bElems):
    copySameShape_cpu(dstB_tv, srcB_tv, 8)
  doAssert xorHash(packB_buf) == bHash_ref, "copySameShape_08 B"
  packB_buf.fill(0)

  bench("copySameShape_16", nb_samples, bElems):
    copySameShape_cpu(dstB_tv, srcB_tv, 16)
  doAssert xorHash(packB_buf) == bHash_ref, "copySameShape_16 B"
  packB_buf.fill(0)

  # ── A-packing ──
  echo "\n-- A packing (", aElems, " floats) --"
  block:
    var buf = newSeq[float32](aElems)
    bench("copyMem (raw)", nb_samples, aElems):
      copyMem(addr buf[0], addr panelA[0], aElems * sizeof(float32))
  copy_flatIndex(dstA_tv, srcA_tv)
  let aHash_ref = xorHash(packA_buf)
  packA_buf.fill(0)

  bench("autoresearch", nb_samples, aElems):
    copy_autoresearch_A(packA_buf, panelA, 1, mc, mpT, kc, mr, mc)
  doAssert xorHash(packA_buf) == aHash_ref, "autoresearch A"
  packA_buf.fill(0)

  bench("laser01 wheel", nb_samples, aElems):
    copy_laser01(packA_buf, aDstSt, panelA, aSrcSt, aShape)
  doAssert xorHash(packA_buf) == aHash_ref, "laser01 A"
  packA_buf.fill(0)

  bench("laser02 strided", nb_samples, aElems):
    copy_laser02(packA_buf, aDstSt, panelA, aSrcSt, aShape)
  doAssert xorHash(packA_buf) == aHash_ref, "laser02 A"
  packA_buf.fill(0)

  bench("laser03 nested for-loops", nb_samples, aElems):
    copy_laser03(packA_buf, aDstSt, panelA, aSrcSt, aShape)
  doAssert xorHash(packA_buf) == aHash_ref, "laser03 A"
  packA_buf.fill(0)

  bench("laser05 fused", nb_samples, aElems):
    copy_laser05(packA_buf, aDstSt, panelA, aSrcSt, aShape)
  doAssert xorHash(packA_buf) == aHash_ref, "laser05 A"
  packA_buf.fill(0)

  bench("crd2idx flat", nb_samples, aElems):
    copy_flatIndex(dstA_tv, srcA_tv)
  doAssert xorHash(packA_buf) == aHash_ref, "flat A"
  packA_buf.fill(0)

  bench("copySameShape_cpu", nb_samples, aElems):
    copySameShape_cpu(dstA_tv, srcA_tv)
  doAssert xorHash(packA_buf) == aHash_ref, "copySameShape_cpu A"
  packA_buf.fill(0)

  bench("copySameShape_08", nb_samples, aElems):
    copySameShape_cpu(dstA_tv, srcA_tv, 8)
  doAssert xorHash(packA_buf) == aHash_ref, "copySameShape_08 A"
  packA_buf.fill(0)

  bench("copySameShape_16", nb_samples, aElems):
    copySameShape_cpu(dstA_tv, srcA_tv, 16)
  doAssert xorHash(packA_buf) == aHash_ref, "copySameShape_16 A"
  packA_buf.fill(0)


  echo ""

  echo "\nAll hash checks passed."

  # ════════════════════════════════════════════════════════════════════
  #  Fused copyMem (contiguity-based) benchmarks
  #  Tests the contiguity-detection logic: contiguous suffix of dims
  #  (matching strides in both src and dst) fuses into a single copyMem.
  # ════════════════════════════════════════════════════════════════════
  echo "\n-- Fused copyMem --"
  echo "  --- 960 floats (L1) ---"
  block:
    # 4D: non-compact src (padded outer), compact dst.
    # src strides (200,30,5,1), dst strides (160,20,5,1).
    # After stride-sort (by dst): order [0,1,2,3], shapes [6,8,4,5].
    #   d=3 (inner): srcSt=1 == innerProd=1 ✓, dstSt=1 == 1 ✓ → fuse
    #   d=2:         srcSt=5 == innerProd=5 ✓, dstSt=5 == 5 ✓ → fuse
    #   d=1:         srcSt=30 != innerProd=20 ✗ → break
    # → last 2 dims fuse (4*5=20 elems/copy), outer loop 6*8=48 iters
    const N_4d = 6*8*4*5
    let bufN = 1230  # max idx = (6-1)*200 + (8-1)*30 + (4-1)*5 + (5-1)*1 + 1
    var pSrc = newSeq[float32](bufN)
    var pDst = newSeq[float32](N_4d)
    for i in 0 ..< bufN: pSrc[i] = rand(1.0'f32)
    let psv = make_view(pSrc, make_layout((6,8,4,5), (200,30,5,1)))
    let pdv = make_view(pDst, make_layout((6,8,4,5), LayoutRight))
    copy_flatIndex(pdv, psv)
    let refH = xorHash(pDst)
    pDst.fill(0)
    var rawBuf = newSeq[float32](N_4d)
    var rawSrc = newSeq[float32](N_4d)
    for i in 0 ..< N_4d: rawSrc[i] = rand(1.0'f32)
    bench("copyMem (raw)", nb_samples, N_4d):
      copyMem(addr rawBuf[0], addr rawSrc[0], N_4d * sizeof(float32))
    bench("copySameShape_padLast2", nb_samples, N_4d):
      copySameShape_cpu(pdv, psv)
    doAssert xorHash(pDst) == refH, "padLast2"
    pDst.fill(0)

  block:
    # 4D: only dim-0 padded on src, compact dst.
    # src strides (200,20,5,1), dst strides (160,20,5,1).
    # After stride-sort: order [0,1,2,3], shapes [6,8,4,5].
    #   d=3: srcSt=1 == 1 ✓, dstSt=1 == 1 ✓ → fuse
    #   d=2: srcSt=5 == 5 ✓, dstSt=5 == 5 ✓ → fuse
    #   d=1: srcSt=20 == 20 ✓, dstSt=20 == 20 ✓ → fuse
    #   d=0: srcSt=200 != 160 ✗ → break
    # → last 3 dims fuse (8*4*5=160 elems/copy), outer 6 iters
    const N_4d2 = 6*8*4*5
    let bufN2 = 1160  # max idx = (6-1)*200 + (8-1)*20 + (4-1)*5 + (5-1)*1 + 1
    var pSrc2 = newSeq[float32](bufN2)
    var pDst2 = newSeq[float32](N_4d2)
    for i in 0 ..< bufN2: pSrc2[i] = rand(1.0'f32)
    let psv2 = make_view(pSrc2, make_layout((6,8,4,5), (200,20,5,1)))
    let pdv2 = make_view(pDst2, make_layout((6,8,4,5), LayoutRight))
    copy_flatIndex(pdv2, psv2)
    let refH2 = xorHash(pDst2)
    pDst2.fill(0)
    bench("copySameShape_padLast3", nb_samples, N_4d2):
      copySameShape_cpu(pdv2, psv2)
    doAssert xorHash(pDst2) == refH2, "padLast3"
    pDst2.fill(0)
  echo "  --- 512000 floats (2 MB) ---"

  block:
    # 3D: both LayoutRight, compact, all dims match → single flat copyMem.
    let N3 = 40*80*160
    var src3 = newSeq[float32](N3)
    var dst3 = newSeq[float32](N3)
    for i in 0 ..< N3: src3[i] = rand(1.0'f32)
    let sv3 = make_view(src3, make_layout((40,80,160), LayoutRight))
    let dv3 = make_view(dst3, make_layout((40,80,160), LayoutRight))
    copy_flatIndex(dv3, sv3)
    let refH3 = xorHash(dst3)
    dst3.fill(0)
    var rawBuf3 = newSeq[float32](N3)
    var rawSrc3 = newSeq[float32](N3)
    for i in 0 ..< N3: rawSrc3[i] = rand(1.0'f32)
    bench("copyMem (raw)", nb_samples, N3):
      copyMem(addr rawBuf3[0], addr rawSrc3[0], N3 * sizeof(float32))
    dst3.fill(0)
    bench("copySameShape_fused3", nb_samples, N3):
      copySameShape_cpu(dv3, sv3)
    doAssert xorHash(dst3) == refH3, "fused3"
    dst3.fill(0)

  block:
    # 3D: both LayoutLeft, compact, all dims match → single flat copyMem.
    let N3l = 40*80*160
    var sl3 = newSeq[float32](N3l)
    var dl3 = newSeq[float32](N3l)
    for i in 0 ..< N3l: sl3[i] = rand(1.0'f32)
    let sv3l = make_view(sl3, make_layout((40,80,160), LayoutLeft))
    let dv3l = make_view(dl3, make_layout((40,80,160), LayoutLeft))
    copy_flatIndex(dv3l, sv3l)
    let refH3l = xorHash(dl3)
    dl3.fill(0)
    bench("copySameShape_fused3_LL", nb_samples, N3l):
      copySameShape_cpu(dv3l, sv3l)
    doAssert xorHash(dl3) == refH3l, "fused3_LL"
    dl3.fill(0)
  echo "  --- 16384 floats (64 KB, permuted) ---"

  block:
    # 4D permute NCHW→CNHW, LayoutRight.
    # src (4,8,16,32): strides (4096,512,32,1)
    # dst (8,4,16,32): strides (2048,512,32,1), perm [1,0,2,3]
    # After stride-sort by dst: order [0,1,2,3], shapes [8,4,16,32].
    # Contiguity: dims 2,3 fuse (16*32=512 elems/copy). Outer loop: 8*4=32 iters.
    const ncHW_N = 4*8*16*32
    var ncHW_src = newSeq[float32](ncHW_N)
    var ncHW_dst = newSeq[float32](ncHW_N)
    for i in 0 ..< ncHW_N: ncHW_src[i] = rand(1.0'f32)
    let ncHW_sv = make_view(ncHW_src, make_layout((4,8,16,32), LayoutRight))
    let ncHW_dv = make_view(ncHW_dst, make_layout((8,4,16,32), LayoutRight))
    copy_flatIndex(ncHW_dv, ncHW_sv)
    let ncHW_ref = xorHash(ncHW_dst)
    ncHW_dst.fill(0)
    var rawBufNC = newSeq[float32](ncHW_N)
    var rawSrcNC = newSeq[float32](ncHW_N)
    for i in 0 ..< ncHW_N: rawSrcNC[i] = rand(1.0'f32)
    bench("copyMem (raw)", nb_samples, ncHW_N):
      copyMem(addr rawBufNC[0], addr rawSrcNC[0], ncHW_N * sizeof(float32))
    bench("copyPermuted_NCtoCN", nb_samples, ncHW_N):
      copyPermuted_cpu(ncHW_dv, ncHW_sv, [1,0,2,3])
    doAssert xorHash(ncHW_dst) == ncHW_ref, "copyPermuted_NCtoCN"
    ncHW_dst.fill(0)
  block:
    # 4D permute NCHW→CNHW, LayoutLeft. No innermost fusion:
    # stride-1 dim is at pos 0 in both layouts, but after perm
    # [1,0,2,3] and stride-sort by dst, the innermost sorted position
    # maps to permDtoS[0]=1 → rawSrcSt[1]=4 ≠ 1. So contiguity fails.
    # Outer dims (512,32) actually match, but the suffix-only walk
    # starts from innermost and breaks before reaching them.
    const ncHW_ll_N = 4*8*16*32
    var ncll_src = newSeq[float32](ncHW_ll_N)
    var ncll_dst = newSeq[float32](ncHW_ll_N)
    for i in 0 ..< ncHW_ll_N: ncll_src[i] = rand(1.0'f32)
    let ncll_sv = make_view(ncll_src, make_layout((4,8,16,32), LayoutLeft))
    let ncll_dv = make_view(ncll_dst, make_layout((8,4,16,32), LayoutLeft))
    copy_flatIndex(ncll_dv, ncll_sv)
    let ncll_ref = xorHash(ncll_dst)
    ncll_dst.fill(0)
    bench("copyPermuted_NCtoCN_LL", nb_samples, ncHW_ll_N):
      copyPermuted_cpu(ncll_dv, ncll_sv, [1,0,2,3])
    doAssert xorHash(ncll_dst) == ncll_ref, "copyPermuted_NCtoCN_LL"
    ncll_dst.fill(0)

  block:
    # LayoutLeft (32,16,8,4) → (32,16,4,8), perm [0,1,3,2].
    # src strides (1,32,512,4096), dst strides (1,32,512,2048).
    # After stride-sort by dst: order [3,2,1,0], shapes [8,4,16,32].
    # Contiguity: dims 2,3 fuse (16*32=512 elems/copy). Outer 8*4=32 iters.
    # This proves LayoutLeft CAN fuse when the perm keeps stride-1 dims
    # at matching stride-sorted positions.
    const ll2_N = 32*16*8*4
    var ll2_src = newSeq[float32](ll2_N)
    var ll2_dst = newSeq[float32](ll2_N)
    for i in 0 ..< ll2_N: ll2_src[i] = rand(1.0'f32)
    let ll2_sv = make_view(ll2_src, make_layout((32,16,8,4), LayoutLeft))
    let ll2_dv = make_view(ll2_dst, make_layout((32,16,4,8), LayoutLeft))
    copy_flatIndex(ll2_dv, ll2_sv)
    let ll2_ref = xorHash(ll2_dst)
    ll2_dst.fill(0)
    bench("copyPermuted_LLkswp", nb_samples, ll2_N):
      copyPermuted_cpu(ll2_dv, ll2_sv, [0,1,3,2])
    doAssert xorHash(ll2_dst) == ll2_ref, "copyPermuted_LLkswp"
    ll2_dst.fill(0)

  proc runTr(NR, NC: static int) =
    echo ""
    echo "-- Transpose ", NR, "×", NC, " (", NR*NC, " floats) --"
    let elems = NR * NC
    var src = newSeq[float32](elems)
    for i in 0 ..< elems:
      src[i] = rand(1.0'f32)
    block:
      var buf = newSeq[float32](elems)
      bench("copyMem (raw)", nb_samples, elems):
        copyMem(addr buf[0], addr src[0], elems * sizeof(float32))
    let sv = make_view(src, make_layout((NR, NC), (1, NR)))

    var refH: uint32
    block:
      var dst = newSeq[float32](elems)
      let dv = make_view(dst, make_layout((NC, NR), (1, NC)))
      copy_flatIndex(dv, sv)
      refH = xorHash(dst)

    block:
      var buf = newSeq[float32](elems)
      let dv = make_view(buf, make_layout((NC, NR), (1, NC)))
      bench("crd2idx flat", nb_samples, elems):
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
      bench("cache_block", nb_samples, elems):
        transpose2D_cacheBlock(addr buf[0], addr src[0], NR, NC)
      doAssert xorHash(buf) == refH
    block:
      var buf = newSeq[float32](elems)
      bench("cache_blk_prefetch", nb_samples, elems):
        transpose2D_cacheBlockPrefetch(addr buf[0], addr src[0], NR, NC)
      doAssert xorHash(buf) == refH
    block:
      var buf = newSeq[float32](elems)
      let dv = make_view(buf, make_layout((NC, NR), (1, NC)))
      bench("copyPermuted_cpu", nb_samples, elems):
        copyPermuted_cpu(dv, sv, [1, 0])
      doAssert xorHash(buf) == refH
    block:
      var buf = newSeq[float32](elems)
      let dv = make_view(buf, make_layout((NC, NR), (1, NC)))
      bench("copyPermuted_16", nb_samples, elems):
        copyPermuted_cpu(dv, sv, [1, 0], 16)
      doAssert xorHash(buf) == refH
  runTr(4000, 2000)
  runTr(2000, 4000)

  echo "\nDone."

when isMainModule:
  main()
