# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Kernel arg delivery: lowered widths, chunk packing, positional streaming
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/crucible/tests/runtime/test_kernel_args.nim

import std/unittest
import workspace/crucible
import workspace/crucible/src/runtime/kernel_args

type
  TestHandle = distinct int
  Whole = object
    a*, b*, c*: int32
    big*: int64
    f*: float32
  WithPtr = object
    p*: ptr uint32
    n*: int32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Kernel args - delivery":

    test "scalars deliver at lowered widths, positional":
      type C = object
        idxs, lens: seq[int]
      proc collect(c: var C, isBuffer: bool, h: TestHandle, idx: int, p: pointer, len: int) {.nimcall.} =
        c.idxs.add idx
        c.lens.add len
      var c: C
      bindArgs(c, bkMetal, 7, (8'u, 1.5'f64, 2'i32, 3'i64), false, collect)
      check c.idxs == @[0, 1, 2, 3, 4]
      check c.lens == @[4, 4, 8, 4, 8]

    test "Nim int delivers truncated to 32 bits":
      type C = object
        buf: array[8, byte]
      proc collect(c: var C, isBuffer: bool, h: TestHandle, idx: int, p: pointer, len: int) {.nimcall.} =
        copyMem(c.buf[idx * 4].addr, p, len)
      var c: C
      bindArgs(c, bkMetal, 7'i32, (0x1234_5678,), false, collect)
      check c.buf == [byte 7, 0, 0, 0, 0x78, 0x56, 0x34, 0x12]

    test "ptr-free composite delivers one chunk at natural alignment":
      type C = object
        lens: seq[int]
        blob: array[32, byte]
      proc collect(c: var C, isBuffer: bool, h: TestHandle, idx: int, p: pointer, len: int) {.nimcall.} =
        c.lens.add len
        if len == 32:
          copyMem(c.blob[0].addr, p, len)
      var c: C
      var w = Whole(a: 1, b: 2, c: 3, big: 8'i64, f: 2.5'f32)
      bindArgs(c, bkMetal, 7'i32, (w,), false, collect)
      check c.lens == @[4, 32]
      check c.blob[12 ..< 16] == [byte 0, 0, 0, 0]      # pad, big aligns to 16
      check c.blob[16 ..< 24] == [byte 8, 0, 0, 0, 0, 0, 0, 0]
      check c.blob[24 ..< 28] == [byte 0, 0, 0x20, 0x40]

    test "buffer-bearing composites explode, handles deliver as buffers":
      type C = object
        kinds: seq[bool]
      proc collect(c: var C, isBuffer: bool, h: TestHandle, idx: int, p: pointer, len: int) {.nimcall.} =
        c.kinds.add isBuffer
      var c: C
      var p = cast[ptr uint32](0xDEAD'i64)
      bindArgs(c, bkMetal, TestHandle(1), (WithPtr(p: p, n: 5),), true, collect)
      check c.kinds == @[true, true, false]

    test "launchArgCount matches the deliveries":
      const n = launchArgCount(int32, tuple[a: Whole, b: int64], TestHandle, bkMetal)
      check n == 3

runTest()
