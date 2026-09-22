# ───────────────  launch_contract (host-side launch-site contract checks)  ───────────────

## Host-side validators for the launch-site contracts of the ceramic tile kernels.
##
## Device procs carry the same contracts in prose:
## - a device-side assert cannot abort a Metal dispatch
## - the host launch site is the only enforcement point
##
## Violations are silent:
## - a wrong copy-path binding or head mapping yields wrong output
## - a bad qScale or an index overflow yields device-side Inf/NaN/OOB
##
## Launch sites call one validator per contract before `engine.run`, see each proc for the call patterns.

import workspace/crucible

const BindingPageSize* = 16384
  ## Metal no-copy binding alignment, the host page size.
  ##
  ## The engine's `newBufferWithBytesNoCopy` path requires it
  ## (see the Metal engine's eligibleNoCopy predicate).

func assertNocopyBinding*[T](arg: PtrArg[T]) {.inline.} =
  ## Asserts the buffer binds through the engine's no-copy path.
  ##
  ## Contract:
  ## - page-aligned pointer
  ## - page-multiple byte length
  ##
  ## Any other binding copies and the kernel's in-place state updates and y
  ## writes are lost silently.
  let nbytes = arg.len * sizeof(T)
  doAssert arg.buf != nil, "no-copy binding: nil buffer"
  doAssert (cast[uint](arg.buf) mod BindingPageSize) == 0,
    "no-copy binding needs a page-aligned pointer"
  doAssert nbytes mod BindingPageSize == 0,
    "no-copy binding needs a page-multiple byte length, got " & $nbytes

func assertNocopyBinding*[T](p: ptr UncheckedArray[T], elems: int) {.inline.} =
  ## Raw-pointer form over a (buf, len) pair, the same contract as the PtrArg form.
  assertNocopyBinding(PtrArg[T](buf: p, len: elems, off: 0))

func assertLanes32*(blkX: SomeInteger) {.inline.} =
  ## Asserts the launch's threadgroup width.
  ##
  ## Contract:
  ## - every tile kernel's lane→element walk, shuffle butterfly and column
  ##   indexing assumes 32-lane threadgroups
  ## - any other width misindexes silently
  doAssert int(blkX) == 32,
    "the tile kernels' lane→element walk is a 32-lane contract, blk.x = " & $blkX

func assertHeadMapping*(Hv, Hk, hkRatio: SomeInteger) {.inline.} =
  ## Asserts the GQA head mapping the value-head→key-head arithmetic assumes.
  ##
  ## Contract:
  ## - `Hv` an exact multiple of `Hk`
  ## - `hkRatio = Hv div Hk` and `hkRatio > 0`
  ##
  ## A zero or wrong ratio reads a wrong or out-of-bounds key head silently.
  doAssert Hk > 0 and Hv > 0, "head mapping needs a non-empty head grid"
  doAssert Hv mod Hk == 0 and hkRatio == Hv div Hk and hkRatio > 0,
    "head mapping needs Hv an exact multiple of Hk and hkRatio = Hv div Hk > 0"

func assertQScale*(qScale: float32) {.inline.} =
  ## Asserts the KDA per-element q̃ divisor.
  ##
  ## Contract:
  ## - the decode and prefill cores divide q by the runtime `qScale`
  ## - qScale is the host's f64 √Dk cast to f32, finite and > 0
  ##
  ## A zero or non-finite scale makes q̃ Inf/NaN and poisons the persistent state.
  doAssert qScale > 0 and qScale != Inf,
    "qScale must be finite and > 0"

func assertPrefillExtent32*(rows, T, dim: SomeInteger) {.inline.} =
  ## Asserts the prefill linear-index bound.
  ##
  ## Contract:
  ## - the in-kernel head/sequence linear bases (`hk·T·Dk`, `bh·T·Dv`)
  ##   and the per-token offsets are int32
  ## - the element count `rows·T·dim` must stay below int32 high
  ##
  ## A larger prefill wraps the indices into out-of-bounds reads and writes.
  let n = int64(rows) * int64(T) * int64(dim)
  doAssert n < int32.high.int64,
    "prefill element count " & $n & " exceeds the int32 linear-index bound"

func assertEpsPositive*(eps: float32) {.inline.} =
  ## Asserts the RMSNorm epsilon.
  ##
  ## Contract:
  ## - eps must be finite and > 0
  ##
  ## An all-zero row with eps = 0 gives rsqrt(0) = +Inf, which the store writes silently.
  doAssert eps > 0 and eps != Inf,
    "the norm epsilon must be finite and > 0"
