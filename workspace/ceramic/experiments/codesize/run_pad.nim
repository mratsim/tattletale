## Codesize ledger for the pad family, comparing the hand-emitted macros
## against the concat-composed recursion candidate over the real call sites.
##
## Every kernel compiles one call site to Metal Shading Language and prints
## its MSL byte size via the crucible metal backend.
##
## - R-suffixed templates = the candidate form, pad to rank through concat
##   plus make_layout recursion with no LayoutCT
## - unsuffixed names = the current src
## - value parity is asserted on the host side before the kernels
##
## Run from the tattletale/ dir with the suite flags of config.nims testerCmd,
## usage in experiments/codesize/README.md.
## Baselines live in the codesize ledger under .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
# ── the candidate form, pad to rank over existing concat ──

func padRightImpl(layout: Layout; n: static int): auto =
  when rank(layout) >= n: layout
  else: padRightImpl(make_layout(concat(layout.shape, 1), concat(layout.stride, 0)), n)

func padLeftImpl(layout: Layout; n: static int): auto =
  when rank(layout) >= n: layout
  else: padLeftImpl(make_layout(concat(1, layout.shape), concat(0, layout.stride)), n)

func padRightR(layout: Layout; rank: static int): auto = padRightImpl(layout, rank)
func padLeftR(layout: Layout; rank: static int): auto = padLeftImpl(layout, rank)

# ── the one-shot candidate, single make_layout with a locally built fill ──
# padRight/padLeft keep their role as the pad-to-rank primitives.
# This test measures whether the fill can be built without the stepwise
# intermediate constructions that pad the recursion route.

template repeatFill(v: Int, n: static int): auto =
  when n <= 0: ()
  else: concat(repeatFill(v, n - 1), v)

func padRightO(layout: Layout; n: static int): auto =
  when rank(layout) >= n: layout
  else: make_layout(concat(layout.shape, repeatFill(Int[1](), n - rank(layout))),
                    concat(layout.stride, repeatFill(Int[0](), n - rank(layout))))

func padLeftO(layout: Layout; n: static int): auto =
  when rank(layout) >= n: layout
  else: make_layout(concat(repeatFill(Int[1](), n - rank(layout)), layout.shape),
                    concat(repeatFill(Int[0](), n - rank(layout)), layout.stride))

# candidate consumers, same bodies as src with the candidate pads
template blockedProductR(blk, tlr): auto =
  const mxR = max(rank(typeof(blk)), rank(typeof(tlr)))
  let lp = logical_product(padRightR(blk, mxR), padRightR(tlr, mxR))
  zipDimensions(dimension(lp, 0), dimension(lp, 1))

template rakedProductR(blk, tlr): auto =
  const mxR = max(rank(typeof(blk)), rank(typeof(tlr)))
  let lp = logical_product(padRightR(blk, mxR), padRightR(tlr, mxR))
  zipDimensions(dimension(lp, 1), dimension(lp, 0))

template tileToShapeR(blk, ts): auto =
  const R = static(rank(ts))
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(tss, ts)
    let padded_blk = padRightR(bk, R)
    let blk_shape = product_each(padded_blk.shape)
    let trg_flat = product_each(tss)
    let product_shape = zipDimensionsWith(trg_flat, blk_shape): ceil_div(it_a, it_b)
    let tiler = make_layout(product_shape, LayoutLeft)
    blockedProductR(padded_blk, tiler)

# ── host-side value parity ──

block:
  let L = make_layout((8, 4), (16, 1))
  let N = make_layout((8, (4, 2)), (16, (1, 2)))
  let S = make_layout(8, 4)
  let P = make_layout((8, 4, 1), (16, 1, 0))
  doAssert padRightR(L, 4) === padRight(L, 4)
  doAssert padLeftR(L, 4) === padLeft(L, 4)
  doAssert padRightR(N, 4) === padRight(N, 4)
  doAssert padLeftR(N, 4) === padLeft(N, 4)
  doAssert padRightR(S, 3) === padRight(S, 3)
  doAssert padLeftR(S, 3) === padLeft(S, 3)
  doAssert padRightR(P, 3) === P
  doAssert padLeftR(P, 3) === P
  doAssert padRightR(P, 2) === P
  let blk = make_layout((8, 4), (16, 1))
  let tiler = make_layout(2, 2)
  doAssert blockedProductR(blk, tiler) === blocked_product(blk, tiler)
  doAssert rakedProductR(blk, tiler) === raked_product(blk, tiler)
  doAssert tileToShapeR(make_layout((2, 3), (1, 2)), (6, 12)) ===
    tile_to_shape(make_layout((2, 3), (1, 2)), (6, 12))
  doAssert padRightO(L, 4) === padRight(L, 4)
  doAssert padLeftO(L, 4) === padLeft(L, 4)
  doAssert padRightO(N, 4) === padRight(N, 4)
  doAssert padLeftO(N, 4) === padLeft(N, 4)
  doAssert padRightO(S, 3) === padRight(S, 3)
  doAssert padLeftO(S, 3) === padLeft(S, 3)
  doAssert padRightO(P, 3) === P
  doAssert padLeftO(P, 3) === P
echo "VALUE PARITY OK"

# ── pad alone ──

const padRightCurrentMsl = metal:
  proc padRightCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padRight(L, 4)
    C[0] = float32 toIntVal size(r)

const padRightRecMsl = metal:
  proc padRightRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padRightR(L, 4)
    C[0] = float32 toIntVal size(r)

const padLeftCurrentMsl = metal:
  proc padLeftCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padLeft(L, 4)
    C[0] = float32 toIntVal size(r)

const padLeftRecMsl = metal:
  proc padLeftRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padLeftR(L, 4)
    C[0] = float32 toIntVal size(r)

const padRightOneShotMsl = metal:
  proc padRightOneShotKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padRightO(L, 4)
    C[0] = float32 toIntVal size(r)

const padLeftOneShotMsl = metal:
  proc padLeftOneShotKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padLeftO(L, 4)
    C[0] = float32 toIntVal size(r)

# ── the consumers, one rank step of pad ──

const blockedCurrentMsl = metal:
  proc blockedCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let blk = make_layout((8, 4), (16, 1))
    let tiler = make_layout(2, 2)
    let r = blocked_product(blk, tiler)
    C[0] = float32 toIntVal size(r)

const blockedRecMsl = metal:
  proc blockedRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let blk = make_layout((8, 4), (16, 1))
    let tiler = make_layout(2, 2)
    let r = blockedProductR(blk, tiler)
    C[0] = float32 toIntVal size(r)

const rakedCurrentMsl = metal:
  proc rakedCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let blk = make_layout((8, 4), (16, 1))
    let tiler = make_layout(2, 2)
    let r = raked_product(blk, tiler)
    C[0] = float32 toIntVal size(r)

const rakedRecMsl = metal:
  proc rakedRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let blk = make_layout((8, 4), (16, 1))
    let tiler = make_layout(2, 2)
    let r = rakedProductR(blk, tiler)
    C[0] = float32 toIntVal size(r)

# ── tile_to_shape, the full consumer chain ──

const tileToShapeCurrentMsl = metal:
  proc tileToShapeCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = tile_to_shape(make_layout((2, 3), (1, 2)), (6, 12))
    C[0] = float32 toIntVal size(r)

const tileToShapeRecMsl = metal:
  proc tileToShapeRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = tileToShapeR(make_layout((2, 3), (1, 2)), (6, 12))
    C[0] = float32 toIntVal size(r)

echo "padRightCurrentKernel: ", cstring(padRightCurrentMsl).len
echo "padRightRecKernel: ", cstring(padRightRecMsl).len
echo "padLeftCurrentKernel: ", cstring(padLeftCurrentMsl).len
echo "padLeftRecKernel: ", cstring(padLeftRecMsl).len
echo "padRightOneShotKernel: ", cstring(padRightOneShotMsl).len
echo "padLeftOneShotKernel: ", cstring(padLeftOneShotMsl).len
echo "blockedCurrentKernel: ", cstring(blockedCurrentMsl).len
echo "blockedRecKernel: ", cstring(blockedRecMsl).len
echo "rakedCurrentKernel: ", cstring(rakedCurrentMsl).len
echo "rakedRecKernel: ", cstring(rakedRecMsl).len
echo "tileToShapeCurrentKernel: ", cstring(tileToShapeCurrentMsl).len
echo "tileToShapeRecKernel: ", cstring(tileToShapeRecMsl).len
