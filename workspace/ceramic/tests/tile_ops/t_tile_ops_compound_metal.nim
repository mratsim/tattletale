# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile compound assignments over Metal: scalar `*=` and the tile forms,
## same-shape (`+=` `-=` `*=` `/=`) and broadcast (`+.= ` `-.=` `*.=` `/.=`).
##
## Operator names must be valid C++ identifiers: the Metal compiler rejects `*=` punctuation, so every launch dies at library compilation.
##
## One threadgroup of 32 threads over a 128-element strip: `a` is the source and `b` a tile of 2s.
## The four tile forms cancel pairwise and the result is 2 * src.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/ceramic/tests/tile_ops/t_tile_ops_compound_metal.nim

import std/[os, strutils]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/tests/ceramic_testutils

export int_tuples, layout_algebra, tensors

proc compoundKernel(dst, src, opSrc: distinct TensorView) =
  let thrID = getLocalThreadID()
  var a: tile(src.T)
  a.load(src, thrID, 0)
  var b: tile(opSrc.T)
  b.load(opSrc, thrID, 0)
  a *= 2'f32
  a += b
  a -= b
  a *= b
  a /= b
  dst.store(a, thrID, 0)

proc broadcastKernel(dst, src, opSrc: distinct TensorView) =
  let thrID = getLocalThreadID()
  var a: tile(src.T)
  a.load(src, thrID, 0)
  var b: tile(opSrc.T)
  b.load(opSrc, thrID, 0)
  a +.= b
  a -.= b
  a *.= b
  a /.= b
  dst.store(a, thrID, 0)

proc compoundCase[E: HwEngine](engine: E, msg: string) =
  var src = makeHostTensor(float32, (1, 128), (128, 1))
  var opSrc = makeHostTensor(float32, (1, 128), (128, 1))
  var got = makeHostTensor(float32, (1, 128), (128, 1))
  var expect = makeHostTensor(float32, (1, 128), (128, 1))
  for k in 0 ..< 128:
    src[0, k] = float32(k) - 64'f32
    opSrc[0, k] = 2'f32
    expect[0, k] = 2'f32 * (float32(k) - 64'f32)
  let
    srcDev = engine.bridge(src)
    opDev = engine.bridge(opSrc)
    gotDev = engine.bridge(got)
  engine.launchDeviceFn compoundKernel(gotDev, srcDev, opDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, got.len())
  let report = allClose(got.view(), expect.view(), rtol = 1e-5, abstol = 1e-4)
  echo msg, ": worst |d| = ", formatFloat(report.worst, ffScientific, 3)
  doAssert report.bad == 0, msg & ": " & $report.bad & " mismatched elements"

proc runTest() =
  compoundCase(newEngine(runMetal), "compound assignments on a 128-element strip tile")

when isMainModule:
  runTest()
