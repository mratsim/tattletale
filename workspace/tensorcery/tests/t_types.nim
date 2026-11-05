# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensorcery type system: erased face over a CPU tensor, rank refusal.
##
## Test that a Tensor created from host data on CPU reads numel, shape, strides, dtype and device through the erased face.
## Test that the typed implementation is recovered from the erased value.
## Test that rank-5 construction raises an AssertionDefect naming rank 5.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --nimcache:nimcache/t_types workspace/tensorcery/tests/t_types.nim

import std/importutils
import std/strutils

import pkg/iface
import workspace/data_structures/src/small_seqs
import workspace/tensorcery

privateAccess(Tensor)
privateAccess(TensorImpl[float32, bkCpu])

proc main() =
  let ctx = Ctx()

  block creation:
    let t = toTensor(ctx, @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
    doAssert t.getNumel() == 6
    doAssert t.getShape() == [2'i32, 3'i32]
    doAssert t.getStrides() == [3'i32, 1'i32]
    doAssert t.getDtype() == dtFloat32
    doAssert t.getDevice() == devCpu

    let back = t.impl.to(TensorImpl[float32, bkCpu])
    doAssert back != nil
    doAssert back.shape == [2'i32, 3'i32]
    doAssert back.strides == [3'i32, 1'i32]
    doAssert back.storage.data == @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0]

  block rank4:
    let t4 = toTensor(ctx, @[1, 2, 3, 4, 5, 6, 7, 8], 1, 2, 2, 2)
    doAssert t4.getNumel() == 8
    doAssert t4.getShape() == [1'i32, 2'i32, 2'i32, 2'i32]
    doAssert t4.getStrides() == [8'i32, 4'i32, 2'i32, 1'i32]
    doAssert t4.getDtype() == dtInt64
    doAssert t4.getDevice() == devCpu

  block rank5Refusal:
    var refusal = ""
    try:
      discard toTensor(ctx, @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0], 1, 1, 1, 1, 2)
    except AssertionDefect as e:
      refusal = e.msg
    doAssert "rank 5" in refusal, "refusal must name the rank, got: " & refusal

main()
