import std/[macros, math, typetraits]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
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

macro isolate*(guide: tuple, d: static int): untyped =
  var builder = TupleBuilderNested.new(1, dropEmpty = false)
  for event in guide.tupleStream():
    builder.onLeaves(event):
      if event.path[^1] == d:
        builder.append(newLit(1))
      else:
        builder.append(newLit(0))
  result = builder.emit(0, emitScalarForSize1 = false)

func size*(t: IntOrIntTuple): auto {.inline.} =
  fold(t, Int[1](), acc * it)

func zero*(T: typedesc): T {.inline.} =
  when T is float16: 0'u16.asFp16()
  elif T is bfloat16: 0'u16.asBf16()
  else: T(0)

template locateTileInSource*(L: Layout, tileShape, tileID: untyped): untyped =
  let tileTiled = zipped_divide(L, tileShape)
  (make_layout(tileTiled.shape[0], tileTiled.stride[0]),
   crd2idx(tileTiled, (0, tileID)))

template partitionTileOnThreads*[ShB, StB, ShM, StM](
    blk: Layout[ShB, StB], threadMap: Layout[ShM, StM], threadIdx: int): untyped =
  let
    atomMap = make_layout(threadMap.shape[0], threadMap.stride[0])
    cellTiled = zipped_divide(blk, (threadMap.stride[1][0],
                                    cosize(atomMap) div threadMap.stride[1][0]))
    tv = compose(cellTiled, (atomMap, X()))
    thrBits = idx2crd(atomMap.shape[0], threadIdx)
    threadCells = tv((thrBits, X()), repeatLike(cellTiled.shape[1], X()))
  (threadCells, crd2idx(tv, threadIdx))

template loadTile*[T, ShD, StD, ShS, StS, ShM, StM](
    dst: var TensorOwned[T, ShD, StD], src: TensorView[T, ShS, StS],
    threadMap: Layout[ShM, StM], threadIdx: int, tileID: IntOrIntTuple,
    padWith = 0) =
  for i in 0 ..< dst.data.len:
    dst.data[i] = padWith
  let tidN = makeIntTuple(tileID)
  let
    cellOuter = threadMap.stride[1][0]
    cellLast = cosize(make_layout(threadMap.shape[0], threadMap.stride[0])) div cellOuter
    tileShape = (cellOuter * threadMap.shape[1][0], cellLast * threadMap.shape[1][1])
    srcShape = src.getLayout().shape
    srcAxis0 = make_layout(srcShape, isolate(srcShape, 0))
    srcAxis1 = make_layout(srcShape, isolate(srcShape, 1))
  let
    (blkS, originS) = locateTileInSource(src.getLayout(), tileShape, tidN)
    (blk0, origin0) = locateTileInSource(srcAxis0, tileShape, tidN)
    (blk1, origin1) = locateTileInSource(srcAxis1, tileShape, tidN)
    (thrS, baseS) = partitionTileOnThreads(blkS, threadMap, threadIdx)
    (thr0, base0) = partitionTileOnThreads(blk0, threadMap, threadIdx)
    (thr1, base1) = partitionTileOnThreads(blk1, threadMap, threadIdx)
  let
    threadCells = make_view(addr src.data[originS + baseS], thrS)
    corner0 = origin0 + base0
    corner1 = origin1 + base1
  for i in 0 ..< dst.data.len:
    if (corner0 + thr0(i), corner1 + thr1(i)).inBounds(srcShape):
      dst.data[i] = threadCells(i)

template storeTile*[T, ShD, StD, ShS, StS, ShM, StM](
    dst: TensorView[T, ShS, StS], src: TensorOwned[T, ShD, StD],
    threadMap: Layout[ShM, StM], threadIdx: int, tileID: IntOrIntTuple) =
  let tidN = makeIntTuple(tileID)
  let
    cellOuter = threadMap.stride[1][0]
    cellLast = cosize(make_layout(threadMap.shape[0], threadMap.stride[0])) div cellOuter
    tileShape = (cellOuter * threadMap.shape[1][0], cellLast * threadMap.shape[1][1])
    dstShape = dst.getLayout().shape
    dstAxis0 = make_layout(dstShape, isolate(dstShape, 0))
    dstAxis1 = make_layout(dstShape, isolate(dstShape, 1))
  let
    (blkS, originS) = locateTileInSource(dst.getLayout(), tileShape, tidN)
    (blk0, origin0) = locateTileInSource(dstAxis0, tileShape, tidN)
    (blk1, origin1) = locateTileInSource(dstAxis1, tileShape, tidN)
    (thrS, baseS) = partitionTileOnThreads(blkS, threadMap, threadIdx)
    (thr0, base0) = partitionTileOnThreads(blk0, threadMap, threadIdx)
    (thr1, base1) = partitionTileOnThreads(blk1, threadMap, threadIdx)
  let
    threadCells = make_view(addr dst.data[originS + baseS], thrS)
    corner0 = origin0 + base0
    corner1 = origin1 + base1
  for i in 0 ..< src.data.len:
    if (corner0 + thr0(i), corner1 + thr1(i)).inBounds(dstShape):
      threadCells(i) = src.data[i]

template get_slice*[ShS, StS, ShM, StM](
    srcLayout: Layout[ShS, StS], threadMap: Layout[ShM, StM],
    threadIdx: int, tileID: IntOrIntTuple): untyped =
  let
    atomMap = make_layout(threadMap.shape[0], threadMap.stride[0])
    cellOuter = threadMap.stride[1][0]
    cellLast = cosize(atomMap) div cellOuter
    tileShape = (cellOuter * threadMap.shape[1][0], cellLast * threadMap.shape[1][1])
    tidN = makeIntTuple(tileID)
    (blk, origin) = locateTileInSource(srcLayout, tileShape, tidN)
    (frg, base) = partitionTileOnThreads(blk, threadMap, threadIdx)
    tileTiled = zipped_divide(make_identity_layout(srcLayout.shape), tileShape)
    cellTiledId = zipped_divide(make_layout(tileTiled.shape[0], tileTiled.stride[0]),
                                (cellOuter, cellLast))
    crd = compose(cellTiledId, (atomMap, X()))
    cellsShape = (crd.shape[0][1], crd.shape[1][0], crd.shape[1][1])
    thrBits = idx2crd(atomMap.shape[0], threadIdx)
    crdOrigin = crd2idx(tileTiled, ((0, 0), tidN))
  (frg, origin + base, thrBits, crdOrigin, crd, cellsShape)

template loadTileIdentity*[T, ShD, StD, ShS, StS, ShM, StM](
    dst: var TensorOwned[T, ShD, StD], src: TensorView[T, ShS, StS],
    threadMap: Layout[ShM, StM], threadIdx: int, tileID: IntOrIntTuple,
    padWith = 0) =
  for i in 0 ..< dst.data.len:
    dst.data[i] = padWith
  let
    (frg, off, thrBits, crdOrigin, crd, cellsShape) =
      get_slice(src.getLayout(), threadMap, threadIdx, tileID)
    srcFrg = make_view(addr src.data[off], frg)
  for i in 0 ..< dst.data.len:
    let (v, r1, r2) = idx2crd(cellsShape, i)
    let crdT = crdOrigin + crd2idx(crd, ((thrBits, v), (r1, r2)))
    if crdT.inBounds(src.getLayout().shape):
      dst.data[i] = srcFrg(i)

template storeTileIdentity*[T, ShD, StD, ShS, StS, ShM, StM](
    dst: TensorView[T, ShS, StS], src: TensorOwned[T, ShD, StD],
    threadMap: Layout[ShM, StM], threadIdx: int, tileID: IntOrIntTuple) =
  let
    (frg, off, thrBits, crdOrigin, crd, cellsShape) =
      get_slice(dst.getLayout(), threadMap, threadIdx, tileID)
    dstFrg = make_view(addr dst.data[off], frg)
  for i in 0 ..< src.data.len:
    let (v, r1, r2) = idx2crd(cellsShape, i)
    let crdT = crdOrigin + crd2idx(crd, ((thrBits, v), (r1, r2)))
    if crdT.inBounds(dst.getLayout().shape):
      dstFrg(i) = src.data[i]

template tileWith*(v: TensorView, tiler: static tuple): auto =
  zipped_divide(v.getLayout(), tiler)

template getLocalTileID*(geom: Layout): auto =
  block:
    let w = threadgroups_per_grid.x
    let h = threadgroups_per_grid.y
    let d = threadgroups_per_grid.z
    let x = threadgroup_position_in_grid.x
    let y = threadgroup_position_in_grid.y
    let z = threadgroup_position_in_grid.z
    let fleet = make_layout(makeIntTuple((w, h, d)))
    crd2idx(fleet, (x, y, z))

template getLocalThreadID*(mma: static TiledMma): int =
  thread_index_in_threadgroup

template gemm*(mma: static TiledMma, C, A, B: untyped) =
  let cGeom = tileWith(C, (mma.MT, mma.NT))
  let aGeom = tileWith(A, (mma.MT, mma.KT))
  let bGeom = tileWith(B, (mma.NT, mma.KT))
  let tileID = getLocalTileID(cGeom)
  let threadIdx = getLocalThreadID(mma)
  let (m, n) = idx2crd(cGeom.shape[1], tileID)
  var rC = rt(C.T, mma.getMmaLayout(opC))
  var rA = rt(A.T, mma.getMmaLayout(opA))
  var rB = rt(B.T, mma.getMmaLayout(opB))
  let aMap = mma.getMatmulThreadmap(opA)
  let bMap = mma.getMatmulThreadmap(opB)
  let cMap = mma.getMatmulThreadmap(opC)
  if tileID < size(cGeom.shape[1]) and threadIdx < mma.atomThreads:
    rC.fillWith(zero(C.T))
    let kTiles = int32(toInt(aGeom.shape[1][1]))
    for k in 0'i32 ..< kTiles:
      rA.loadTile(A, aMap, threadIdx, (int32(m), int32(k)))
      rB.loadTile(B, bMap, threadIdx, (int32(n), int32(k)))
      mma.atom.gemm_tile(rC, rA, rB)
    C.storeTile(rC, cMap, threadIdx, (int32(m), int32(n)))
