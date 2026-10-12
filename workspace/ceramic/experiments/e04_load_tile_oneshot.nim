import std/[macros, math, typetraits]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/crucible
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu
import workspace/ceramic/src/macros/static_for

proc getMmaAtom*(TAcc, TA, TB: typedesc): static MmaAtom {.inline.} =
  static:
    doAssert TAcc is float32, "the picked atoms accumulate in fp32"
  when ccGetBackend() == ctMetal:
    static:
      doAssert TA is TB, "the Metal simdgroup atoms require equal operand types"
    when TA is float16: APPLE_8x8x8_F16
    elif TA is float32: APPLE_8x8x8_F32
    elif TA is bfloat16: APPLE_8x8x8_BF16
    else: {.error: "no Metal MMA atom for the requested element types".}
  else:
    when TAcc is float32 and TA is float16 and TB is float16:
      UNIVERSAL_1x1x1_F32F16F16F32
    elif TAcc is float32 and TA is float32 and TB is float32:
      UNIVERSAL_1x1x1_F32F32F32F32
    elif TAcc is float32 and TA is bfloat16 and TB is bfloat16:
      UNIVERSAL_1x1x1_F32BF16BF16F32
    else: {.error: "no MMA atom for the requested element types".}

type TiledMma*[TAcc, TA, TB; MT, NT, KT: static int] = object
  atom*: MmaAtom

proc tiledMma*[TAcc, TA, TB](
      _: typedesc[TAcc], _: typedesc[TA], _: typedesc[TB],
      MT, NT, KT: static int
    ): TiledMma[TAcc, TA, TB, MT, NT, KT] {.inline.} =
  const atom = getMmaAtom(TAcc, TA, TB)
  when MT mod atom.getM() == 0 and NT mod atom.getN() == 0 and KT mod atom.getK() == 0:
    result = TiledMma[TAcc, TA, TB, MT, NT, KT](atom: atom)
  else:
    {.error: "the tile dimensions are not multiples of the resolved atom's cell dimensions".}

template getMmaLayout*(mma: static TiledMma, op: static MmaOperand): untyped =
  const
    vpt = mma.atom.valuesPerThread(op)
    cellOuter = (when op == opB: mma.atom.getN() else: mma.atom.getM())
    cellLast = (when op == opC: mma.atom.getN() else: mma.atom.getK())
    repOuter = (when op == opB: mma.NT else: mma.MT) div cellOuter
    repLast = (when op == opC: mma.NT else: mma.KT) div cellLast
  make_layout((vpt, (repOuter, repLast)))

template getMatmulThreadmap*(mma: static TiledMma, op: static MmaOperand): untyped =
  const
    atomMap = mma.atom.getLayout(op)
    cellOuter = (when op == opB: mma.atom.getN() else: mma.atom.getM())
    cellLast = (when op == opC: mma.atom.getN() else: mma.atom.getK())
    repOuter = (when op == opB: mma.NT else: mma.MT) div cellOuter
    repLast = (when op == opC: mma.NT else: mma.KT) div cellLast
    tileOuter = cellOuter * repOuter
  make_layout((atomMap.shape, (repOuter, repLast)),
              (atomMap.stride, (cellOuter, tileOuter * cellLast)))

template atomThreads*(mma: static TiledMma): int =
  toInt(mma.atom.threadCount(opA))

template rt*(T: typedesc, dims: IntOrIntTuple): untyped =
  make_tensor(T, makeIntTuple(dims))

template rt*(T: typedesc, L: Layout): untyped =
  make_tensor(T, L)

macro repeatLike*(guide: tuple, elem: untyped): untyped =
  var builder = TupleBuilderNested.new(1, dropEmpty = false)
  for event in guide.tupleStream():
    builder.onLeaves(event):
      builder.append(elem)
  result = builder.emit(0, emitScalarForSize1 = false)

template size*(t: IntOrIntTuple): auto =
  product(t)

template zero*(T: typedesc): untyped =
  when T is float16: cast[T](0'u16)
  elif T is bfloat16: cast[T](0'u16)
  else: T(0)

type ThreadValueMap* = distinct Layout

template atomShape*(tv: ThreadValueMap): untyped = tv.Layout.shape[0][0]

template atomsPerTile*(tv: ThreadValueMap): untyped = tv.Layout.shape[0][1]

template threadGrid*(tv: ThreadValueMap): untyped = tv.Layout.shape[1]

template getTVCrdLayout*(mma: static TiledMma, op: static MmaOperand): untyped =
  const threadMap = mma.getMatmulThreadmap(op)
  const tileShape = ((when op == opB: mma.NT else: mma.MT),
                     (when op == opC: mma.NT else: mma.KT))
  ThreadValueMap(compose(make_identity_layout(tileShape), threadMap))

proc loadTile*[T, ShD, StD, ShS, StS](
    dst: var TensorOwned[T, ShD, StD], src: TensorView[T, ShS, StS],
    tvcrd: static ThreadValueMap, threadIdx: int, tileID: (int, int),
    padWith: T = zero(T)) =
  let srcLayout = src.getLayout()
  let R = zipped_divide(srcLayout, tvcrd.Layout)
  let thrBits = idx2crd(tvcrd.atomShape, threadIdx)
  let threadGrid = tvcrd.threadGrid
  let sel = (((thrBits, _), repeatLike(threadGrid, _)), tileID)
  let srcReg = src.slice(R, sel)
  let reg = srcReg.getLayout()
  let RC = zipped_divide(make_identity_layout(srcLayout.shape), tvcrd.Layout)
  let regC = slice(RC, sel)
  let srcShape = srcLayout.shape
  for i in 0 ..< size(reg):
    if crd2idx(regC, i).inBounds(srcShape):
      dst[i] = srcReg[i]
    else:
      dst[i] = padWith

proc storeTile*[T, ShD, StD, ShS, StS](
    dst: TensorView[T, ShD, StD], src: TensorOwned[T, ShS, StS],
    tvcrd: static ThreadValueMap, threadIdx: int, tileID: (int, int)) =
  let dstLayout = dst.getLayout()
  let R = zipped_divide(dstLayout, tvcrd.Layout)
  let thrBits = idx2crd(tvcrd.atomShape, threadIdx)
  let threadGrid = tvcrd.threadGrid
  let sel = (((thrBits, _), repeatLike(threadGrid, _)), tileID)
  let dstReg = dst.slice(R, sel)
  let reg = dstReg.getLayout()
  let RC = zipped_divide(make_identity_layout(dstLayout.shape), tvcrd.Layout)
  let regC = slice(RC, sel)
  let dstShape = dstLayout.shape
  for i in 0 ..< size(reg):
    if crd2idx(regC, i).inBounds(dstShape):
      dstReg[i] = src[i]

template gemm*(mma: static TiledMma, C, A, B: untyped, tileID: untyped, threadIdx: untyped) =
  const tvcrdA = mma.getTVCrdLayout(opA)
  const tvcrdB = mma.getTVCrdLayout(opB)
  const tvcrdC = mma.getTVCrdLayout(opC)
  let cGeom = zipped_divide(C.getLayout(), (mma.MT, mma.NT))
  let aGeom = zipped_divide(A.getLayout(), (mma.MT, mma.KT))
  let tid = toInt(tileID)
  let thr = toInt(threadIdx)
  if tid < size(cGeom.shape[1]) and thr < mma.atomThreads:
    let (m, n) = idx2crd(cGeom.shape[1], tid)
    let kTiles = toInt(aGeom.shape[1][1])
    const cLayout = mma.getMmaLayout(opC)
    var rC = rt(C.T, cLayout)
    var rA = rt(A.T, mma.getMmaLayout(opA))
    var rB = rt(B.T, mma.getMmaLayout(opB))
    for i in 0 ..< size(cLayout.shape):
      rC[i] = zero(C.T)
    for k in 0 ..< kTiles:
      rA.loadTile(A, tvcrdA, thr, (m, k))
      rB.loadTile(B, tvcrdB, thr, (n, k))
      mma.atom.gemm_tile(rC, rA, rB)
    C.storeTile(rC, tvcrdC, thr, (m, n))
