# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout-algebra conformance against the frozen python reference implementation.
##
## The same cases as python/tests/test_partition_identity.py.
##
## Test that the partition identity, proven in python, reproduces in Nim
## across the layouts the passes choose:
##
##   composition(zipped_divide(D, tiler), (tv, None)) ==
##     zipped_divide(D, composition(tiler, tv))
##
## The ownership tables and the reference layout dumps match the vendored
## pycute reference byte for byte.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/b4 \
##     --outdir:build/b4 workspace/tattletile/tests/test_layout_algebra_conformance.nim

import std/algorithm
import std/strutils

import workspace/tattletile/src/tattletile/layout_algebra

proc sm70Tv(): Layout =
  ## SM70 8x8x4 MMA thread-value layout from the verified pycute test suite
  makeLayout(it(it(2, 2, 2), it(2, 2, 2)), it(it(1, 16, 4), it(8, 2, 32)))

proc checkIdentity(D: Layout, tiler: IntTuple, tv: Layout, label: string) =
  ## Asserts both partition forms are equal and cover every element of D
  let twoStep = composeModes(zippedDivide(D, tiler), @[someMode(tv), noneMode()])
  let oneStep = zippedDivide(D, compose(tiler, tv))
  doAssert twoStep == oneStep, label & ": two-step " & $twoStep &
    " != one-step " & $oneStep
  doAssert size(twoStep) == size(D),
    label & ": partition size " & $size(twoStep) & " != data size " & $size(D)

proc testIdentitySm70MmaLayout() =
  ## SM70 8x8x4 MMA associativity pair, data in both stride orders
  let tilerMn = it(8, 8)
  checkIdentity(makeLayout(it(32, 64)), tilerMn, sm70Tv(), "sm70 row-major")
  checkIdentity(makeLayout(it(32, 64), it(64, 1)), tilerMn, sm70Tv(), "sm70 col-major")

proc testIdentityAxpyTile() =
  ## Coalesced tv layout the axpy pass picks: 4 threads, 2 values each
  let D = makeLayout(itv(8), itv(1))
  let tiler = makeLayout(itv(8), itv(1))
  let tv = makeLayout(it(4, 2), it(1, 4))
  checkIdentity(D, tiler.shape, tv, "axpy tile")
  # exact partition form, not shape-and-stride equality alone
  doAssert $zippedDivide(D, compose(tiler.shape, tv)) ==
    "Layout(((4, 2), 1), ((1, 4), 8))", $zippedDivide(D, compose(tiler.shape, tv))

proc testIdentityGemmAccumulator() =
  ## Coalesced tv layout the GEMM pass picks over the 16x16 accumulator
  let D = makeLayout(it(16, 16), it(16, 1))
  let tv = makeLayout(it(4, 64), it(1, 4))
  checkIdentity(D, it(16, 16), tv, "gemm accumulator")

proc testAxpyOwnershipIsCoalesced() =
  ## Thread t owns elements {t, t + threads}: consecutive threads, adjacent elements, the tile covered exactly once.
  let tv = makeLayout(it(4, 2), it(1, 4))
  var owned: seq[seq[int]] = @[]
  for t in 0 ..< 4:
    var row: seq[int] = @[]
    for v in 0 ..< 2:
      row.add tv.callInt(it(t, v))
      doAssert tv.callInt(it(t, v)) == tv.callInt(t + v * 4),
        "flat and natural coordinate forms agree"
    owned.add row
  doAssert owned == @[@[0, 4], @[1, 5], @[2, 6], @[3, 7]],
    "thread t owns {t, t + threads}, got " & $owned
  var flat: seq[int] = @[]
  for row in owned:
    for x in row:
      flat.add x
  flat.sort()
  doAssert flat == @[0, 1, 2, 3, 4, 5, 6, 7],
    "ownership must cover the tile exactly once"

proc testSizeIsTheLeafProduct() =
  ## Size of a layout is the product of its shape leaves, real values:
  ## the partition identity asserts would pass vacuously on a zero size
  doAssert size(makeLayout(it(8, 64), it(64, 1))) == 512, "8x64 tile"
  doAssert size(makeLayout(it(32, 64))) == 2048, "sm70 data tile"
  doAssert size(sm70Tv()) == 64, "sm70 tv layout: 32 threads x 2 values"
  doAssert size(makeLayout(itv(8), itv(1))) == 8, "axpy tile"

proc testReferenceLayoutDumps() =
  ## Reference implementation intermediate layouts, byte for byte
  block: # sm70, implicit strides
    let D = makeLayout(it(32, 64))
    let tv = sm70Tv()
    doAssert $tilerToLayout(it(8, 8)) == "Layout((8, 8), (1@0, 1@1))",
      $tilerToLayout(it(8, 8))
    doAssert $zippedDivide(D, it(8, 8)) ==
      "Layout(((8, 8), (4, 8)), ((1, 32), (8, 256)))", $zippedDivide(D, it(8, 8))
    doAssert $compose(it(8, 8), tv) ==
      "Layout(((2, 2, 2), (2, 2, 2)), ((1@0, 2@1, 4@0), (1@1, 2@0, 4@1)))",
      $compose(it(8, 8), tv)
    let twoStep = composeModes(zippedDivide(D, it(8, 8)), @[someMode(tv), noneMode()])
    doAssert $twoStep ==
      "Layout((((2, 2, 2), (2, 2, 2)), (4, 8)), (((1, 64, 4), (32, 2, 128)), (8, 256)))",
      $twoStep
    doAssert strT(cosize(twoStep)) == "2048", strT(cosize(twoStep))
    doAssert strT(cosize(D)) == "2048", strT(cosize(D))
  block: # sm70 column-major
    let Dcm = makeLayout(it(32, 64), it(64, 1))
    let tv = sm70Tv()
    doAssert $zippedDivide(Dcm, it(8, 8)) ==
      "Layout(((8, 8), (4, 8)), ((64, 1), (512, 8)))", $zippedDivide(Dcm, it(8, 8))
    let twoStep = composeModes(zippedDivide(Dcm, it(8, 8)), @[someMode(tv), noneMode()])
    doAssert $twoStep ==
      "Layout((((2, 2, 2), (2, 2, 2)), (4, 8)), (((64, 2, 256), (1, 128, 4)), (512, 8)))",
      $twoStep
  block: # gemm accumulator
    let D = makeLayout(it(16, 16), it(16, 1))
    let tv = makeLayout(it(4, 64), it(1, 4))
    doAssert $compose(it(16, 16), tv) == "Layout((4, (4, 16)), (1@0, (4@0, 1@1)))",
      $compose(it(16, 16), tv)
    let two = composeModes(zippedDivide(D, it(16, 16)), @[someMode(tv), noneMode()])
    doAssert $two == "Layout(((4, (4, 16)), (1, 1)), ((16, (64, 1)), (256, 16)))",
      $two
  block: # gemm tv ownership flats of thread 0
    let g = makeLayout(it(4, 64), it(1, 4))
    var row: seq[int] = @[]
    for v in 0 ..< 8:
      row.add g.callInt(it(0, v))
    doAssert row == @[0, 4, 8, 12, 16, 20, 24, 28], $row

proc testAlgebraOpsMatchReference() =
  ## Single-op sanity values produced by the reference implementation
  doAssert $complement(makeLayout(it(4), itv(2))) == "Layout((2, 1), (1, 8))",
    $complement(makeLayout(it(4), itv(2)))
  doAssert $complement(makeLayout(itv(8), itv(1)), it(32)) == "Layout(4, 8)",
    $complement(makeLayout(itv(8), itv(1)), it(32))
  doAssert $rightInverse(makeLayout(it(4, 8), it(1, 4))) == "Layout(32, 1)",
    $rightInverse(makeLayout(it(4, 8), it(1, 4)))
  doAssert $rightInverse(makeLayout(it(4, 8), it(8, 1))) == "Layout((8, 4), (4, 1))",
    $rightInverse(makeLayout(it(4, 8), it(8, 1)))
  doAssert $coalesce(makeLayout(it(2, 1, 6, 1), it(1, 7, 8, 0))) ==
    "Layout((2, 6), (1, 8))", $coalesce(makeLayout(it(2, 1, 6, 1), it(1, 7, 8, 0)))
  doAssert $logicalProduct(makeLayout(it(2, 2), it(4, 1)), itv(6)) ==
    "Layout(((2, 2), (2, 3)), ((4, 1), (2, 8)))",
    $logicalProduct(makeLayout(it(2, 2), it(4, 1)), itv(6))
  doAssert $concatLayout(@[makeLayout(it(3), itv(1)), makeLayout(it(2), itv(42))]) ==
    "Layout((3, 2), (1, 42))"

proc refusesLayout(body: proc, detail: string) =
  ## Runs `body` and requires a LayoutError naming `detail` in the message
  try:
    body()
    doAssert false, "expected a LayoutError naming '" & detail & "'"
  except LayoutError as e:
    doAssert detail in e.msg, "unexpected LayoutError: " & e.msg

proc testGeometryRefusals() =
  ## Test that inputs outside the algebra's domain refuse loudly:
  ## an integer coordinate over an empty shape, a non-positive shape leaf,
  ## a non-positive tile size, a negative stride under cosize
  refusesLayout(proc() = discard callInt(concatLayout(@[]), 0),
                "integer coordinate over an empty shape")
  refusesLayout(proc() = discard callInt(makeLayout(it(0, 2), it(1, 1)), 1),
                "non-positive shape leaf")
  refusesLayout(proc() = discard logicalDivide(makeLayout(itv(8), itv(1)), itv(0)),
                "non-positive tile size")
  refusesLayout(proc() = discard cosize(makeLayout(it(4), itv(-2))),
                "negative stride")
  doAssert ssZero(ssInt(0)), "the integer 0 is the additive identity"
  doAssert ssZero(ssAtup(@[ssInt(0), ssInt(0)])),
    "an all-zero tuple is the additive identity, leaf by leaf"
  doAssert not ssZero(ssAtup(@[ssInt(0), ssInt(1)])),
    "a nonzero leaf makes the tuple nonzero"

when isMainModule:
  testSizeIsTheLeafProduct()
  testIdentitySm70MmaLayout()
  testIdentityAxpyTile()
  testIdentityGemmAccumulator()
  testAxpyOwnershipIsCoalesced()
  testReferenceLayoutDumps()
  testAlgebraOpsMatchReference()
  testGeometryRefusals()
  echo "[OK] layout algebra conformance: 4 identity checks + ownership tables + reference dumps"
