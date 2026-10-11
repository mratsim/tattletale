import std/[strformat, math]
import workspace/crucible
import workspace/crucible/src/runtime/exec/metal_runtime
import workspace/crucible/src/runtime/engines/arg_blobs
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import ./e02_load_tile_mma
{.experimental: "callOperator".}
func fp32ToFp16(x: float32): uint16 =
  let u = cast[uint32](x)
  let sign = uint16((u shr 16) and 0x8000'u32)
  let aexp = int32((u shr 23) and 0xFF)
  let mant = u and 0x7FFFFF'u32
  if aexp == 0xFF:
    if mant == 0: return sign or 0x7C00'u16
    return sign or 0x7C00'u16 or 0x0200'u16
  var bexp = aexp - 127 + 15
  if bexp >= 31:
    return sign or 0x7C00'u16
  if bexp <= 0:
    if bexp < -10: return sign
    let m2 = mant or 0x800000'u32
    let shift = uint32(126 - aexp)
    let half = 1'u32 shl (shift - 1)
    var r = m2 shr shift
    let rem = m2 and ((1'u32 shl shift) - 1)
    if rem > half or (rem == half and (r and 1) == 1):
      r += 1
    return sign or uint16(r)
  let half = 0x1000'u32
  var r = mant shr 13
  let rem = mant and 0x1FFF'u32
  if rem > half or (rem == half and (r and 1) == 1):
    r += 1
    if r == 0x400:
      bexp += 1
      r = 0
      if bexp >= 31:
        return sign or 0x7C00'u16
  return sign or uint16(bexp shl 10) or uint16(r)
func fp16ToFp32(h: uint16): float32 =
  let sign = uint32(h and 0x8000'u16) shl 16
  let e = (h shr 10) and 0x1F'u16
  let m = h and 0x3FF'u16
  if e == 0:
    if m == 0: return cast[float32](sign)
    var m2 = uint32(m)
    var e2 = 113
    while (m2 and 0x400'u32) == 0:
      m2 = m2 shl 1
      dec e2
    return cast[float32](sign or (uint32(e2) shl 23) or ((m2 and 0x3FF'u32) shl 13))
  if e == 0x1F:
    return cast[float32](sign or 0x7F800000'u32 or (uint32(m) shl 13))
  return cast[float32](sign or (uint32(e + 112) shl 23) or (uint32(m) shl 13))
const gemmMsl = metal:
  proc gemmK(D: TensorView[float32, (int, int), (int, int)],
             A: TensorView[float16, (int, int), (int, int)],
             B: TensorView[float16, (int, int), (int, int)]) {.global.} =
    gemm(tiledMma(float32, float16, float16, 32, 32, 16), D, A, B)
proc checkMatmul(engine: HwEngine, metalCtx: MetalCtx, which, label: string,
    M, N, K: int, sA, sB, sC: (int, int), gxMul = 1, gyMul = 1): bool =
  let
    gridX = ((M + 31) div 32) * gxMul
    gridY = ((N + 31) div 32) * gyMul
  var
    dBuf = allocBuffer(metalCtx.device, M * N * sizeof(float32))
    aBuf = allocBuffer(metalCtx.device, M * K * sizeof(uint16))
    bBuf = allocBuffer(metalCtx.device, K * N * sizeof(uint16))
  let
    A = cast[ptr UncheckedArray[uint16]](aBuf.data)
    B = cast[ptr UncheckedArray[uint16]](bBuf.data)
    C = cast[ptr UncheckedArray[float32]](dBuf.data)
  for m in 0 ..< M:
    for k in 0 ..< K:
      A[m * K + k] = fp32ToFp16(float32(1 + (2 * m + 7 * k) mod 13))
  for k in 0 ..< K:
    for n in 0 ..< N:
      B[k * N + n] = fp32ToFp16(float32(1 + (3 * k + 11 * n) mod 17))
  let
    aView = make_view(cast[ptr UncheckedArray[float16]](aBuf.data),
                      makeIntTuple((M, K)),
                      makeIntTuple((sA[0], sA[1])))
    bView = make_view(cast[ptr UncheckedArray[float16]](bBuf.data),
                      makeIntTuple((N, K)),
                      makeIntTuple((sB[0], sB[1])))
    cView = make_view(cast[ptr UncheckedArray[float32]](dBuf.data),
                      makeIntTuple((M, N)),
                      makeIntTuple((sC[0], sC[1])))
  var dArg = PtrArg[float32](buf: cast[ptr UncheckedArray[float32]](dBuf.data), len: M * N, off: 0)
  engine.run << (grid: (gridX, gridY), blk: (32, 1)) >>
    (which, dArg,
      (cView.getLayout(),
       PtrArg[float16](buf: aView.data, len: M * K, off: 0), aView.getLayout(),
       PtrArg[float16](buf: bView.data, len: K * N, off: 0), bView.getLayout()))
  var bad = 0
  var worst = 0.0'f32
  var first = ""
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += fp16ToFp32(A[m * sA[0] + k * sA[1]]) * fp16ToFp32(B[n * sB[0] + k * sB[1]])
      let want = acc
      let got = C[m * sC[0] + n * sC[1]]
      let d = abs(got - want)
      worst = max(worst, d)
      if d > 1e-3'f32:
        inc bad
        if bad == 1: first = &"({m},{n}): got {got} want {want}"
  let pass = bad == 0
  echo (if pass: &"  {label} {M}x{N}x{K} A{sA} B{sB} C{sC}: PASS worst |d| = {worst}"
        else: &"  {label} {M}x{N}x{K} A{sA} B{sB} C{sC}: FAIL {bad} bad, e.g. {first}, worst {worst}")
  releaseBuffer(dBuf)
  releaseBuffer(aBuf)
  releaseBuffer(bBuf)
  return pass
proc main =
  var engine = bkMetal.init()
  engine.ingest(gemmMsl)
  let metalCtx = initMetal()
  var ok = true
  for sA in [(32, 1), (1, 32)]:
    for sB in [(1, 32), (32, 1)]:
      for sC in [(32, 1), (1, 32)]:
        ok = engine.checkMatmul(metalCtx, "gemmK", "gemm", 32, 32, 32, sA, sB, sC) and ok
  ok = engine.checkMatmul(metalCtx, "gemmK", "gemm", 32, 32, 41, (41, 1), (1, 32), (32, 1)) and ok
  ok = engine.checkMatmul(metalCtx, "gemmK", "gemm", 40, 32, 32, (32, 1), (1, 32), (32, 1)) and ok
  ok = engine.checkMatmul(metalCtx, "gemmK", "gemm", 32, 45, 32, (32, 1), (1, 45), (45, 1)) and ok
  ok = engine.checkMatmul(metalCtx, "gemmK", "gemm", 40, 45, 41, (41, 1), (1, 45), (45, 1)) and ok
  ok = engine.checkMatmul(metalCtx, "gemmK", "gemm", 70, 50, 100, (100, 1), (1, 50), (50, 1)) and ok
  ok = engine.checkMatmul(metalCtx, "gemmK", "gemm", 70, 50, 96, (96, 1), (1, 50), (50, 1)) and ok
  ok = engine.checkMatmul(metalCtx, "gemmK", "gemm", 64, 64, 64, (64, 1), (1, 64), (64, 1), gxMul = 3, gyMul = 2) and ok
  ok = engine.checkMatmul(metalCtx, "gemmK", "gemm", 70, 50, 100, (100, 1), (1, 50), (50, 1), gxMul = 2, gyMul = 3) and ok
  if ok: echo "e02 check_gemm: all green"
  else: quit 1
when isMainModule:
  main()
