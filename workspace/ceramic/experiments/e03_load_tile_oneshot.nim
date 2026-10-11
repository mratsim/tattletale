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
export int_tuples, layout_algebra, tensors, h_mma_configgen, h_mma_registry,
       h_mma_properties, h_mma_dispatch, k_layout_gemm, k_layout_fillwith_gpu
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
  fold(t, Int[1](), acc * it)
template zero*(T: typedesc): untyped =
  when T is float16: cast[T](0'u16)
  elif T is bfloat16: cast[T](0'u16)
  else: T(0)
template getTVCrdLayout*(mma: static TiledMma, op: static MmaOperand): untyped =
  const threadMap = mma.getMatmulThreadmap(op)
  const cellOuter = (when op == opB: mma.atom.getN() else: mma.atom.getM())
  const cellLast = (when op == opC: mma.atom.getN() else: mma.atom.getK())
  const tileShape = ((when op == opB: mma.NT else: mma.MT),
                     (when op == opC: mma.NT else: mma.KT))
  const atomMap = threadMap.selectDimensions(0)
  const atomCrd = compose(make_identity_layout((cellOuter, cellLast)), atomMap)
  const atomT = compose(make_layout(tileShape, (1, tileShape[0])), atomCrd)
  const tiledTile = zipped_divide(make_layout(tileShape), (cellOuter, cellLast))
  const cellMap = tiledTile.selectDimensions(1)
  const threadMapT = make_layout((atomT.shape, cellMap.shape),
                                 (atomT.stride, cellMap.stride))
  const tvcrd = compose(make_identity_layout(tileShape), threadMapT)
  const lift = (ism_coord_strides.E(toInt(tileShape[0]), 0),
                ism_coord_strides.E(toInt(tileShape[1]), 1))
  (tvcrd, tileShape, lift)
macro evalTileCoord*(layout: typed, coords: typed): untyped =
  var ty = layout.getTypeInst()
  var st: NimNode
  if ty.kind == nnkBracketExpr and ty.len == 3:
    st = ty[2]
  else:
    let impl = ty.getImpl
    doAssert impl[2].kind == nnkRecList
    var shN: NimNode
    pickShapeStride(impl[2], shN, st)
  var coordLeaves: seq[NimNode] = @[]
  proc collectLeaves(n: NimNode) =
    if n.kind in {nnkTupleConstr, nnkPar}:
      for i in 0 ..< n.len: collectLeaves(n[i])
    else:
      coordLeaves.add n
  collectLeaves(coords)
  var acc: seq[NimNode] = @[]
  proc addTerm(mode: int, term: NimNode) =
    while acc.len <= mode: acc.add newLit(0)
    acc[mode] = nnkInfix.newTree(bindSym"+", acc[mode], term)
  var ci = 0
  for ev in st.tupleStream():
    if ev.kind != kLeaf:
      continue
    let x = coordLeaves[ci]
    inc ci
    let desc = getCoordStrideDescriptor(ev.leafTy)
    if desc.kind == caNone:
      error "evalTileCoord: stride leaf is not a CoordStride", ev.leaf
    var flat: seq[int] = @[]
    proc collect(n: NimNode) =
      if n.kind in {nnkTupleConstr, nnkPar}:
        for i in 0 ..< n.len: collect(n[i])
      else:
        flat.add n.intVal.int
    collect(desc.coeffs)
    for k in 0 ..< flat.len:
      if flat[k] != 0:
        addTerm(k, nnkInfix.newTree(bindSym"*", newLit(flat[k]), x))
  result = if acc.len == 1: acc[0] else: nnkPar.newTree(acc)
template divideByTVCrd*(world, tvcrd, tileShape: untyped): untyped =
  let Rtv = compose(world, tvcrd)
  let tiled = zipped_divide(world, tileShape)
  make_layout((tvcrd.shape, tiled.shape[1]), (Rtv.stride, tiled.stride[1]))
template loadTileOneStep*[T, ShD, StD, ShS, StS](
    dst: var TensorOwned[T, ShD, StD], src: TensorView[T, ShS, StS],
    cfg: untyped, threadIdx: untyped, tileID: IntOrIntTuple,
    padWith: untyped = 0) =
  let tvcrd = cfg[0]
  let tileShape = cfg[1]
  let world = src.getLayout()
  let R = divideByTVCrd(world, tvcrd, tileShape)
  let tidN = (toInt(tileID[0]), toInt(tileID[1]))
  let thrBits = idx2crd(tvcrd.shape[0][0], threadIdx)
  let cellsShape = tvcrd.shape[1]
  let sel = (((thrBits, X()), repeatLike(cellsShape, X())), tidN)
  let fragWrapped = slice(R, sel)
  let frag = fragWrapped.selectDimensions(0)
  let fragOff = crd2idx(R, sel)
  let tileGrid = ((toInt(world.shape[0]) + toInt(tileShape[0]) - 1) div toInt(tileShape[0]),
                  (toInt(world.shape[1]) + toInt(tileShape[1]) - 1) div toInt(tileShape[1]))
  let RC = make_layout((tvcrd.shape, tileGrid), (tvcrd.stride, cfg[2]))
  let pad = (when padWith is int: zero(T) else: padWith)
  for i in 0 ..< size(frag.shape):
    dst.data[i] = pad
  let worldShape = world.shape
  for i in 0 ..< size(frag.shape):
    let (vpt, cellC) = idx2crd(frag.shape, i)
    let worldC = evalTileCoord(RC,
      ((((thrBits[0], thrBits[1], thrBits[2], thrBits[3], thrBits[4]), vpt),
        (cellC[0], cellC[1])), (tidN[0], tidN[1])))
    if worldC.inBounds(worldShape):
      dst.data[i] = src.data[fragOff + crd2idx(frag, i)]
template storeTileOneStep*[T, ShD, StD, ShS, StS](
    dst: TensorView[T, ShD, StD], src: TensorOwned[T, ShS, StS],
    cfg: untyped, threadIdx: int, tileID: IntOrIntTuple) =
  let tvcrd = cfg[0]
  let tileShape = cfg[1]
  let world = dst.getLayout()
  let R = divideByTVCrd(world, tvcrd, tileShape)
  let tidN = (toInt(tileID[0]), toInt(tileID[1]))
  let thrBits = idx2crd(tvcrd.shape[0][0], threadIdx)
  let cellsShape = tvcrd.shape[1]
  let sel = (((thrBits, X()), repeatLike(cellsShape, X())), tidN)
  let fragWrapped = slice(R, sel)
  let frag = fragWrapped.selectDimensions(0)
  let fragOff = crd2idx(R, sel)
  let tileGrid = ((toInt(world.shape[0]) + toInt(tileShape[0]) - 1) div toInt(tileShape[0]),
                  (toInt(world.shape[1]) + toInt(tileShape[1]) - 1) div toInt(tileShape[1]))
  let RC = make_layout((tvcrd.shape, tileGrid), (tvcrd.stride, cfg[2]))
  let worldShape = world.shape
  for i in 0 ..< size(frag.shape):
    let (vpt, cellC) = idx2crd(frag.shape, i)
    let worldC = evalTileCoord(RC,
      ((((thrBits[0], thrBits[1], thrBits[2], thrBits[3], thrBits[4]), vpt),
        (cellC[0], cellC[1])), (tidN[0], tidN[1])))
    if worldC.inBounds(worldShape):
      dst.data[fragOff + crd2idx(frag, i)] = src.data[i]
template gemm*(mma: static TiledMma, C, A, B: untyped, tileID: untyped, threadIdx: untyped) =
  const cfgA = mma.getTVCrdLayout(opA)
  const cfgB = mma.getTVCrdLayout(opB)
  const cfgC = mma.getTVCrdLayout(opC)
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
      rC.data[i] = zero(C.T)
    for k in 0 ..< kTiles:
      rA.loadTileOneStep(A, cfgA, thr, (m, k))
      rB.loadTileOneStep(B, cfgB, thr, (n, k))
      mma.atom.gemm_tile(rC, rA, rB)
    C.storeTileOneStep(rC, cfgC, thr, (m, n))
