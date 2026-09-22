# ───────────────  launch_contract (host-side launch-site contract checks)  ───────────────

## Host-side validators for the launch-site contracts of the ceramic tile kernels.
##
## Device procs carry the same contracts in prose:
## - a device-side assert cannot abort a Metal dispatch
## - the host launch site is the only enforcement point
##
## Violations are silent:
## - a wrong copy-path binding or head mapping yields wrong output
## - a bad qScale, a bad decay operand or an index overflow yields device-side Inf/NaN/OOB
##
## Launch sites call one validator per contract before `engine.run`, see each proc for the call patterns.

import workspace/crucible

const LaneWidth* = 32
  ## Ceramic tile kernels' launch geometry threadgroup width,
  ## the lane→element walk's contract.

const BindingPageSize* = 16384
  ## Metal no-copy binding alignment, the host page size.
  ##
  ## The engine's `newBufferWithBytesNoCopy` path requires it
  ## (see the Metal engine's eligibleNoCopy predicate).

func assertNocopyBinding*[T](arg: PtrArg[T]) {.inline.} =
  ## Asserts the buffer binds through the engine's no-copy path.
  ##
  ## Contract:
  ## - the engine binds the blob at `buf + off·sizeof(T)` (arg_blobs.blobOf),
  ##   so the binding base, not `buf` alone, must be page-aligned
  ## - page-multiple byte length `len·sizeof(T)`
  ## - every current launch site passes `off = 0`, the assert then reduces
  ##   to `buf`'s own alignment
  ##
  ## Any other binding copies and the kernel's in-place state updates and y
  ## writes are lost silently.
  let nbytes = arg.len * sizeof(T)
  let base = cast[uint](arg.buf) + uint(arg.off * sizeof(T))
  doAssert arg.buf != nil, "no-copy binding: nil buffer"
  doAssert base mod BindingPageSize == 0,
    "no-copy binding needs a page-aligned binding base (buf + off·sizeof(T))"
  doAssert nbytes mod BindingPageSize == 0,
    "no-copy binding needs a page-multiple byte length, got " & $nbytes

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

func assertDecayFinite*(decay: PtrArg[float32], n: int) {.inline.} =
  ## Asserts the log-decay operand contract (finite and ≤ 0 everywhere).
  ##
  ## Contract:
  ## - the operand is a per-token log decay `g`, or the host-computed
  ##   per-channel cumulative log decay `cumg` prefix built from it,
  ##   both ≤ 0 by construction
  ## - the kernel applies no clamp, the operand is consumed as-is
  ## - a > 0 entry turns the exp/exp2 decay factors into growth, a non-finite
  ##   entry into Inf/NaN, the violating values carry into the unrounded
  ##   persistent f32 state
  ##
  ## Example:
  ##   GDN launches `assertDecayFinite(gPA, bhMax)` right before `engine.run`.
  for i in 0 ..< n:
    let d = decay.buf[i]
    doAssert d == d and d != Inf and d != NegInf,
      "log-decay operand must be finite, entry " & $i & " is not"
    doAssert d <= 0.0'f32,
      "log-decay operand must be ≤ 0, entry " & $i & " is " & $d

func assertCumgFinite*(cumg: PtrArg[float32], heads, tokens, channels, chunkLen: int) {.inline.} =
  ## Asserts the per-channel cumulative log decay contract.
  ##
  ## Contract:
  ## - every entry finite and ≤ 0, `assertDecayFinite`'s operand contract
  ## - monotone non-increasing along the token axis inside a chunk, the prefix
  ##   restarts at every chunk boundary
  ## - a rising cumg makes the pairdecay exponent positive, its exp2 overflows
  ##   and the violating values carry into the unrounded persistent f32 state
  ##
  ## The view is `(heads, tokens, channels)` row-major with `channels` (Dk)
  ## the per-token stride, monotonicity stays within one head per chunk.
  ##
  ## Example:
  ##   KDA prefill launches `assertCumgFinite(cumgPA, qkRows, T, Dk, ChunkC)`
  ##   right before `engine.run`.
  let n = heads * tokens * channels
  for i in 0 ..< n:
    let d = cumg.buf[i]
    doAssert d == d and d != Inf and d != NegInf,
      "cumg operand must be finite, entry " & $i & " is not"
    doAssert d <= 0.0'f32,
      "cumg operand must be ≤ 0, entry " & $i & " is " & $d

  for h in 0 ..< heads:
    let base = h * tokens * channels
    for c in 0 ..< channels:
      for c0 in countup(0, tokens - 1, chunkLen):
        var prev = cumg.buf[base + c0 * channels + c]
        for t in c0 + 1 ..< min(c0 + chunkLen, tokens):
          let d = cumg.buf[base + t * channels + c]
          doAssert d <= prev,
            "cumg must be monotone non-increasing within the chunk, entry " & $d
          prev = d
