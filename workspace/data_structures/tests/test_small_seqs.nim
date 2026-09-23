# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run from the repo root, artifacts outside the tree:
##   nim c --path:. --nimcache:/tmp/x --outdir:/tmp/x -r workspace/data_structures/tests/test_small_seqs.nim
##
## Pass `-d:nimAllocStats` to enable the allocation-count assertions.
## Pass `--checks:off` or `-d:danger` to compile the negative bounds posture.

import std/importutils
import workspace/data_structures/small_seqs
privateAccess(SmallSeq)

const EmptyInt32: array[0, int32] = []
  # An empty `array[0, T]` converts to `openArray[T]`.

# ─── Layout ──────────────────────────────────────────────────────────────────
#
# Figures of this platform, arm64 with Nim 2.2.10.

type
  WideSeq = object
    # `SmallSeq` with `int` lengths, the layout the 40-byte figure rejects.
    len: int
    cap: int
    arr: array[5, int32]
    overflow: ptr UncheckedArray[int32]

  NarrowWideSeq = object
    # The same pair of widths at the smaller inline capacity.
    len: int
    cap: int
    arr: array[3, int32]
    overflow: ptr UncheckedArray[int32]

  Aligned = object
    # 8-byte aligned payload, one `int32` then one pointer.
    id: int32
    p: ptr int32

proc layoutChecks =
  doAssert sizeof(SmallSeq[5, int32]) == 40
  doAssert sizeof(SmallSeq[3, int32]) == 32
  doAssert offsetOf(SmallSeq[5, int32], len) == 0
  doAssert offsetOf(SmallSeq[5, int32], cap) == 4
  doAssert offsetOf(SmallSeq[5, int32], arr) == 8
  doAssert offsetOf(SmallSeq[5, int32], overflow) == 32
  doAssert offsetOf(SmallSeq[3, int32], len) == 0
  doAssert offsetOf(SmallSeq[3, int32], cap) == 4
  doAssert offsetOf(SmallSeq[3, int32], arr) == 8
  doAssert offsetOf(SmallSeq[3, int32], overflow) == 24
  doAssert sizeof(ptr UncheckedArray[int32]) == 8
  doAssert sizeof(seq[int32]) == 16
  doAssert sizeof(WideSeq) == 48
  doAssert sizeof(NarrowWideSeq) == 40
  doAssert sizeof(SmallSeq[3, int32]) < sizeof(NarrowWideSeq)
  # An 8-byte element pads the pair out to two cache lines.
  doAssert alignOf(Aligned) == 8
  doAssert offsetOf(SmallSeq[3, Aligned], arr) == 8
  doAssert offsetOf(SmallSeq[3, Aligned], overflow) == 56
  doAssert sizeof(SmallSeq[3, Aligned]) == 64
  doAssert sizeof(SmallSeq[5, Aligned]) == 96

# ─── The index split ─────────────────────────────────────────────────────────

proc prefixNeverMoves =
  # The first `N` elements stay in `arr` however far the sequence grows.
  var s = SmallSeq[5, int32].init
  for i in 0 ..< 5:
    s.add int32(i)
  let firstAddr = addr(s[0])
  doAssert firstAddr == addr(s.arr[0])
  s.add 5'i32
  doAssert s.overflow != nil
  doAssert addr(s[0]) == firstAddr
  for i in 6 ..< 300:
    s.add int32(i)
  doAssert s.len == 300
  doAssert addr(s[0]) == firstAddr
  doAssert addr(s[0]) == addr(s.arr[0])
  doAssert addr(s[4]) == addr(s.arr[4])
  doAssert addr(s[5]) == addr(s.overflow[0])
  doAssert addr(s[299]) == addr(s.overflow[294])
  for i in 0 ..< 300:
    doAssert s[i] == int32(i)

proc boundaryChecks =
  const N = 5
  var s = SmallSeq[5, int32].init
  for i in 0 ..< 8:
    s.add int32(i)
  doAssert s[N - 1] == 4'i32
  doAssert s[N] == 5'i32
  doAssert s[N + 1] == 6'i32
  doAssert addr(s[N - 1]) == addr(s.arr[N - 1])
  doAssert addr(s[N]) == addr(s.overflow[0])
  doAssert addr(s[N + 1]) == addr(s.overflow[1])
  s[N - 1] = 40'i32
  s[N] = 50'i32
  s[N + 1] = 60'i32
  doAssert s.arr[N - 1] == 40'i32
  doAssert s.overflow[0] == 50'i32
  doAssert s.overflow[1] == 60'i32
  doAssert s[N - 1] == 40'i32
  doAssert s[N] == 50'i32
  doAssert s[N + 1] == 60'i32
  s[N - 1] += 1
  s[N] += 1
  doAssert s.arr[N - 1] == 41'i32
  doAssert s.overflow[0] == 51'i32

# ─── Bounds checks ───────────────────────────────────────────────────────────

proc boundsCheckTests =
  # Indexed access follows the compile-time `boundChecks` switch.
  var s = SmallSeq[3, int32].init
  for i in 0 ..< 5:
    s.add int32(i)
  when compileOption("boundChecks"):
    var raisedRead = false
    try:
      discard s[7]
    except IndexDefect:
      raisedRead = true
    doAssert raisedRead
    var raisedWrite = false
    try:
      s[7] = 1'i32
    except IndexDefect:
      raisedWrite = true
    doAssert raisedWrite
    doAssert s[2] == 2'i32
    s[2] = 22'i32
    doAssert s[2] == 22'i32
  when not compileOption("boundChecks"):
    # The tail comes from realloc0, so a slot past `len` reads back as 0.
    var raised = false
    try:
      doAssert s[5] == 0'i32
    except IndexDefect:
      raised = true
    doAssert not raised
    s[5] = 9'i32
    doAssert s[5] == 9'i32

# ─── Correctness ─────────────────────────────────────────────────────────────

proc inlineChecks =
  var s = SmallSeq[5, int32].init
  doAssert s.len == 0
  doAssert len(s) == 0
  doAssert s.overflow == nil
  for i in 0 ..< 5:
    s.add int32(i * i)
  doAssert s.len == 5
  doAssert s.cap == 5
  doAssert s.overflow == nil
  doAssert s == [0'i32, 1, 4, 9, 16]

proc zeroValueChecks =
  # A default-constructed value has `cap` 0, the first spill must still size the tail.
  var s: SmallSeq[3, int32]
  doAssert s.len == 0
  doAssert s.cap == 0
  doAssert s.overflow == nil
  doAssert s == EmptyInt32
  var expected = 0'i32
  for i in 0 ..< 20:
    s.add int32(i * 3)
    expected += int32(i * 3)
  doAssert s.cap == 24
  doAssert s.overflow != nil
  var fromInit = SmallSeq[3, int32].init
  for i in 0 ..< 20:
    fromInit.add int32(i * 3)
  # The zero value and `init` reach the same capacity, the tail grows identically.
  doAssert s.cap == fromInit.cap
  doAssert addr(s[0]) == addr(s.arr[0])
  doAssert addr(s[3]) == addr(s.overflow[0])
  for i in 0 ..< 20:
    doAssert s[i] == int32(i * 3)
    doAssert s[i] == fromInit[i]
  var count = 0
  var sumVal = 0'i32
  for v in items(s):
    sumVal += v
    inc count
  doAssert count == 20
  doAssert sumVal == expected
  var indexed = 0
  for i, v in pairs(s):
    doAssert v == s[i]
    inc indexed
  doAssert indexed == 20
  s[1] = 11'i32
  s[5] = 55'i32
  doAssert s.arr[1] == 11'i32
  doAssert s.overflow[2] == 55'i32
  doAssert s[1] == 11'i32
  doAssert s[5] == 55'i32
  # The copy owns a tail of its own, writing it cannot reach the source.
  var dup = s
  doAssert dup.len == 20
  doAssert dup.cap == s.cap
  doAssert dup.overflow != s.overflow
  dup[0] = 7'i32
  dup[19] = 77'i32
  doAssert s[0] == 0'i32
  doAssert s[19] == 57'i32
  doAssert dup[0] == 7'i32
  doAssert dup[19] == 77'i32
  block:
    # The destructor runs on a value whose tail came from a zero value.
    var dropped: SmallSeq[5, int32]
    for i in 0 ..< 12:
      dropped.add int32(i)
    doAssert dropped.cap == 20
    doAssert dropped.overflow != nil
  block:
    # A value that never appended has nothing to release.
    var untouched: SmallSeq[5, int32]
    untouched.clear()
    doAssert untouched.len == 0
    doAssert untouched.overflow == nil

proc growthChecks =
  var s = SmallSeq[5, int32].init
  for i in 0 ..< 10:
    s.add int32(i)
  doAssert s.cap == 10
  for i in 10 ..< 20:
    s.add int32(i)
  doAssert s.cap == 20
  for i in 20 ..< 40:
    s.add int32(i)
  doAssert s.cap == 40
  doAssert s.len == 40
  for i in 0 ..< 40:
    doAssert s[i] == int32(i)

proc mutationChecks =
  var s = SmallSeq[3, int32].init
  for i in 0 ..< 2:
    s.add int32(i)
  s[1] = 42'i32
  doAssert s[1] == 42'i32
  for v in mitems(s):
    v += 1
  doAssert s == [1'i32, 43]
  for i in 0 ..< 20:
    s.add int32(i)
  doAssert s.len == 22
  s[21] = 7'i32
  doAssert s[21] == 7
  doAssert s.overflow[18] == 7
  var seen = 0
  for v in mitems(s):
    v = 0
    inc seen
  doAssert seen == 22
  let want = newSeq[int32](22)
  doAssert s == want

proc iterationChecks =
  var s = SmallSeq[3, int32].init
  for i in 0 ..< 7:
    s.add int32(i * 3)
  var sumIdx = 0
  var sumVal = 0'i32
  var count = 0
  for i, v in pairs(s):
    sumIdx += i
    sumVal += v
    inc count
  doAssert count == 7
  doAssert sumIdx == 0 + 1 + 2 + 3 + 4 + 5 + 6
  doAssert sumVal == 0 + 3 + 6 + 9 + 12 + 15 + 18
  for i, v in mpairs(s):
    v = int32(i)
  doAssert s == [0'i32, 1, 2, 3, 4, 5, 6]
  var read = 0'i32
  for v in items(s):
    read += v
  doAssert read == 0 + 1 + 2 + 3 + 4 + 5 + 6

proc equalityChecks =
  var s = SmallSeq[4, int32].init
  doAssert s == EmptyInt32
  doAssert not (s == [0'i32])
  for i in 0 ..< 4:
    s.add int32(i)
  doAssert s == [0'i32, 1, 2, 3]
  doAssert not (s == [0'i32, 1, 2, 3, 4])
  doAssert not (s == [0'i32, 1, 2])
  doAssert not (s == [0'i32, 1, 2, 4])
  s.add 4'i32
  doAssert s == [0'i32, 1, 2, 3, 4]
  doAssert not (s == [0'i32, 1, 2, 3])
  var t = SmallSeq[3, int32].init
  for i in 0 ..< 6:
    t.add int32(i)
  doAssert t == [0'i32, 1, 2, 3, 4, 5]
  doAssert not (t == [0'i32, 1, 2, 3, 4, 6])
  doAssert not (t == [0'i32, 1, 2, 3, 4])

proc clearChecks =
  var s = SmallSeq[5, int32].init
  for i in 0 ..< 12:
    s.add int32(i)
  let tail = s.overflow
  doAssert tail != nil
  s.clear()
  doAssert s.len == 0
  doAssert s.overflow == tail
  doAssert s.cap == 20
  doAssert s == EmptyInt32
  for i in 0 ..< 3:
    s.add int32(i)
  doAssert s.overflow == tail
  doAssert s == [0'i32, 1, 2]
  # Growth reallocs the tail, so block identity is taken after the growth.
  for i in 3 ..< 30:
    s.add int32(i)
  let grown = s.overflow
  doAssert s.cap == 40
  s.clear()
  doAssert s.len == 0
  doAssert s.overflow == grown
  # A copied value with a retained tail and `len` 0 copies no elements.
  var emptied = s
  doAssert emptied.len == 0
  doAssert emptied.overflow != nil
  doAssert emptied == EmptyInt32
  var untouched = SmallSeq[5, int32].init
  untouched.clear()
  doAssert untouched.len == 0
  doAssert untouched.overflow == nil

# ─── Allocation counts ───────────────────────────────────────────────────────
#
# `getAllocStats()` counts `alloc`, `alloc0` and `dealloc` calls while
# `realloc0` carries no increment.

when defined(nimAllocStats):
  privateAccess(AllocStats)

  proc counts(a: AllocStats): (int, int) =
    (a.allocCount, a.deallocCount)

  proc allocChecks =
    block:
      var warm = SmallSeq[5, int32].init
      for i in 0 ..< 40:
        warm.add int32(i)
      warm.clear()

    block:
      let before = getAllocStats()
      var s = SmallSeq[5, int32].init
      for i in 0 ..< 5:
        s.add int32(i)
      doAssert s.overflow == nil
      let measured = counts(getAllocStats() - before)
      echo "fill to N: ", measured
      doAssert measured == (0, 0), $measured

    # A 40-element row owns one block, the growths cost no separate allocation.
    block:
      let before = getAllocStats()
      block:
        var s = SmallSeq[5, int32].init
        for i in 0 ..< 40:
          s.add int32(i)
        doAssert s.cap == 40
      let measured = counts(getAllocStats() - before)
      echo "row of 40 dropped: ", measured
      doAssert measured == (0, 1), $measured

    block:
      var s = SmallSeq[5, int32].init
      for i in 0 ..< 40:
        s.add int32(i)
      let tail = s.overflow
      let before = getAllocStats()
      s.clear()
      doAssert s.len == 0
      doAssert s.overflow == tail
      for i in 0 ..< 40:
        s.add int32(i)
      doAssert s.overflow == tail
      let measured = counts(getAllocStats() - before)
      echo "clear and refill to 40: ", measured
      doAssert measured == (0, 0), $measured

    # 1000 refills cost no allocator call, the bound is one fill's growth.
    block:
      var s = SmallSeq[5, int32].init
      for i in 0 ..< 20:
        s.add int32(i)
      let tail = s.overflow
      let before = getAllocStats()
      for n in 0 ..< 1000:
        s.clear()
        for i in 0 ..< 20:
          s.add int32(i)
      doAssert s.len == 20
      doAssert s.overflow == tail
      let measured = counts(getAllocStats() - before)
      echo "1000 refills to 20: ", measured
      doAssert measured == (0, 0), $measured

    # The shifted rows cost nothing, each new row costs one alloc and one free.
    block:
      var rows: seq[SmallSeq[5, int32]] = @[]
      for r in 0 ..< 16:
        var row = SmallSeq[5, int32].init
        for c in 0 ..< 8:
          row.add int32(c + r)
        rows.add move(row)
      let before = getAllocStats()
      for r in 0 ..< 16:
        var row = SmallSeq[5, int32].init
        for c in 0 ..< 8:
          row.add int32(c + r)
        rows.insert(row, 0)
        # The source stays live after the call, so Nim copies it, no move.
        doAssert row.len == 8
      let measured = counts(getAllocStats() - before)
      echo "16 spilled rows inserted at 0: ", measured
      doAssert measured == (16, 16), $measured
      var wrong = 0
      for r in 0 ..< 32:
        # Last inserted row at 0, the original 16 keep their order at 16 ..< 32.
        let src = if r < 16: 15 - r else: r - 16
        for c in 0 ..< 8:
          if rows[r][c] != int32(c + src):
            inc wrong
      doAssert wrong == 0, "contents corrupted by the shift: " & $wrong
      rows.setLen 0

    # A moved source transfers its tail pointer, so the delta stays empty.
    block:
      var rows: seq[SmallSeq[5, int32]] = @[]
      for r in 0 ..< 16:
        var row = SmallSeq[5, int32].init
        for c in 0 ..< 8:
          row.add int32(c + r)
        rows.add move(row)
      let before = getAllocStats()
      for r in 0 ..< 16:
        var row = SmallSeq[5, int32].init
        for c in 0 ..< 8:
          row.add int32(c + r)
        rows.insert(move(row), 0)
      let measured = counts(getAllocStats() - before)
      echo "16 spilled rows moved in at 0: ", measured
      doAssert measured == (0, 0), $measured
      rows.setLen 0

    block:
      var row = SmallSeq[5, int32].init
      for c in 0 ..< 8:
        row.add int32(c)
      let before = getAllocStats()
      var dup = row
      doAssert dup.len == 8
      doAssert dup.overflow != row.overflow
      let measured = counts(getAllocStats() - before)
      echo "one deep copy of a spilled row: ", measured
      doAssert measured == (1, 0), $measured

    block:
      var s = SmallSeq[5, int32].init
      for i in 0 ..< 64:
        s.add int32(i)
      let before = getAllocStats()
      var acc = 0'i32
      for n in 0 ..< 4096:
        acc += s[n mod 64]
        s[n mod 64] = int32(n mod 64)
      let measured = counts(getAllocStats() - before)
      echo "4096 indexed reads and writes: ", measured
      doAssert measured == (0, 0), $measured
      doAssert acc != 0

    # A zero value reaches the same block count as `init`, the tail is one realloc chain.
    block:
      let before = getAllocStats()
      block:
        var s: SmallSeq[5, int32]
        for i in 0 ..< 40:
          s.add int32(i)
        doAssert s.cap == 40
      let measured = counts(getAllocStats() - before)
      echo "zero value row of 40 dropped: ", measured
      doAssert measured == (0, 1), $measured

    block:
      let before = getAllocStats()
      block:
        var s: SmallSeq[5, int32]
        for i in 0 ..< 5:
          s.add int32(i)
        doAssert s.overflow == nil
      let measured = counts(getAllocStats() - before)
      echo "zero value filled to N and dropped: ", measured
      doAssert measured == (0, 0), $measured

    block:
      let before = getAllocStats()
      block:
        var s: SmallSeq[5, int32]
      let measured = counts(getAllocStats() - before)
      echo "zero value never appended to, dropped: ", measured
      doAssert measured == (0, 0), $measured

    block:
      var row: SmallSeq[5, int32]
      for c in 0 ..< 8:
        row.add int32(c)
      let before = getAllocStats()
      var dup = row
      doAssert dup.len == 8
      doAssert dup.overflow != row.overflow
      let measured = counts(getAllocStats() - before)
      echo "deep copy of a zero-value row: ", measured
      doAssert measured == (1, 0), $measured
else:
  proc allocChecks =
    echo "allocator block counts skipped: compile with -d:nimAllocStats to run them"

# ─── Non-trivial elements ────────────────────────────────────────────────────
#
# `string` elements, the hooks working element-wise,
# every dead slot staying zeroed, no copy or sink touching a destroyed element.

proc strChecks =
  var s: SmallSeq[3, string]
  for i in 0 ..< 8:
    s.add "x" & $i
  doAssert s.len == 8
  doAssert s.overflow != nil
  doAssert s[5] == "x5"
  # A copy dups every live element, the two payloads staying independent.
  var d = s
  d[5] = "changed"
  doAssert s[5] == "x5"
  doAssert d[5] == "changed"
  # Sink moves the live elements out, the source left empty and droppable.
  var m: SmallSeq[3, string]
  m = move s
  doAssert m.len == 8
  doAssert m[7] == "x7"
  doAssert s.len == 0
  doAssert m == @["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]
  # Clear destroys the elements and zeroes their slots, a refill then reads clean.
  m.clear()
  doAssert m.len == 0
  m.add "fresh"
  doAssert m[0] == "fresh"
  doAssert m.arr[1] == ""
  # A copy over a non-empty destination replaces every slot, live or dead,
  # the source's dead slots arriving zero, never stale.
  var r: SmallSeq[3, string]
  r.add "a"
  r.add "b"
  r = d
  doAssert r.len == 8
  doAssert r[1] == "x1"
  doAssert r[2] == "x2"
  # A zero value sinks and copies clean.
  var z: SmallSeq[3, string]
  var z2: SmallSeq[3, string]
  z2 = move z
  z2 = SmallSeq[3, string].init
  doAssert z2.len == 0
  doAssert z2.overflow == nil
  # A spilled four-element row, allocation counts measured:
  # - growth, one block alloc
  # - a copy, the block plus one dup per live element, three inline and one overflowed, and one dealloc from the measured rebind
  # - a sink, the block alone, the elements moving
  when defined(nimAllocStats):
    privateAccess(AllocStats)
    proc counts(a: AllocStats): (int, int) =
      (a.allocCount, a.deallocCount)
    var els: array[4, string]
    for i in 0 ..< 4:
      els[i] = "e" & $i
    block:
      let before = getAllocStats()
      var s4: SmallSeq[3, string]
      for i in 0 ..< 4:
        s4.add move els[i]
      doAssert s4.len == 4
      doAssert s4[3] == "e3"
      let measured = counts(getAllocStats() - before)
      echo "spill growth to 4 strings: ", measured
      doAssert measured == (1, 0), $measured
      let beforeCopy = getAllocStats()
      var dup = s4
      doAssert dup.overflow != s4.overflow
      doAssert dup[3] == "e3"
      dup[3] = "other"
      doAssert s4[3] == "e3"
      let measuredCopy = counts(getAllocStats() - beforeCopy)
      echo "deep copy of a spilled 4-string row: ", measuredCopy
      doAssert measuredCopy == (5, 1), $measuredCopy
      var taken: SmallSeq[3, string]
      taken = move s4
      doAssert taken.len == 4
      doAssert taken[3] == "e3"
      doAssert s4.len == 0

proc spillSlotHygiene =
  ## A spilled row's fresh tail block arrives with its dead slots zeroed.
  ## A sink that first fills a dead slot therefore never destroys stale bytes
  ## left from an earlier life.
  ##
  ## Heap pollution precedes the sink, the recycled block replacing memory
  ## the allocator hands back out.
  var junk = newSeq[string]()
  for i in 0 ..< 200:
    junk.add newString(64)
    junk[^1][0] = 'x'
  junk.setLen(0)
  var s: SmallSeq[3, string]
  for i in 0 ..< 4:
    s.add "e" & $i
  doAssert s[3] == "e3"
  # Sink over a spilled destination, the source's live elements moving into
  # a tail block whose dead slots are zero.
  var src: SmallSeq[3, string]
  for i in 0 ..< 5:
    src.add "f" & $i
  s = move src
  doAssert s.len == 5
  doAssert s[4] == "f4"
  # Copy over a spilled destination, the copy building a fresh tail block
  # independent of the source's.
  var c = s
  doAssert c[4] == "f4"
  c[4] = "g4"
  doAssert s[4] == "f4"
  c.clear()
  echo "spill slot hygiene ok"

proc polluteHeap =
  # Recycled blocks of the shape the hooks allocate, one 48-byte spill
  # block per row, its live slot holding a freed string payload pointer.
  var junk: seq[SmallSeq[3, string]]
  for k in 0 ..< 100:
    var row: SmallSeq[3, string]
    for i in 0 ..< 4:
      row.add "j" & $k
    junk.add row
  junk = newSeq[SmallSeq[3, string]](0)

proc sinkRefillHygiene =
  ## Sink, growth and copy all place live elements into a fresh tail block.
  ## An element assignment destroys the slot's prior content first, so a fresh
  ## block must arrive fully zeroed before any element loop runs.
  ##
  ## Zeroing only the dead tail afterwards still lets the first assignment
  ## destroy stale heap bytes.
  ##
  ## Heap pollution of the matching block size stands in for the allocator's
  ## recycled memory, the zero page hiding the bug otherwise.
  polluteHeap()
  # Sink refill over a spilled destination, the fresh tail block taking
  # the source's overflow elements through sink assignments.
  var s: SmallSeq[3, string]
  for i in 0 ..< 4:
    s.add "e" & $i
  var src: SmallSeq[3, string]
  for i in 0 ..< 5:
    src.add "f" & $i
  s = move src
  doAssert s[3] == "f3" and s[4] == "f4"
  # Copy refill over a spilled destination, the fresh tail block taking
  # the source's overflow elements through plain assignments.
  var c = s
  doAssert c[3] == "f3" and c[4] == "f4"
  doAssert s[3] == "f3"
  # Growth move over a live spilled row, the fresh block taking the row's
  # elements through sink assignments.
  polluteHeap()
  var g: SmallSeq[3, string]
  for i in 0 ..< 4:
    g.add "g" & $i
  for i in 4 ..< 7:
    g.add "g" & $i
  doAssert g[6] == "g6" and g[0] == "g0"
  # A spilled short source, its fresh tail block taking no elements at all,
  # the later adds then sinking into the block's dead slots.
  polluteHeap()
  var short: SmallSeq[3, string]
  for i in 0 ..< 4:
    short.add "s" & $i
  var dst: SmallSeq[3, string]
  for i in 0 ..< 4:
    dst.add "d" & $i
  dst = move short
  doAssert dst.len == 4 and dst[3] == "s3"
  dst.add "x"
  dst.add "y"
  doAssert dst[4] == "x" and dst[5] == "y"
  echo "sink refill hygiene ok"

proc main =
  layoutChecks()
  strChecks()
  prefixNeverMoves()
  boundaryChecks()
  boundsCheckTests()
  inlineChecks()
  zeroValueChecks()
  growthChecks()
  sinkRefillHygiene()
  mutationChecks()
  iterationChecks()
  equalityChecks()
  clearChecks()
  spillSlotHygiene()
  allocChecks()
  echo "small_seq: layout, index split, retention and move checks passed"

main()
