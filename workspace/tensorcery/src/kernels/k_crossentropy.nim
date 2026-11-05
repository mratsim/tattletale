# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Cross entropy kernel: one loss per set of v logits, logsumexp + target gather.
##
## For element r of the loss:
##   lse     = max_c logits[r, c] + log(sum_c exp(logits[r, c] - max))
##   loss[r] = lse - logits[r, target[r]]
##
## - The logits may be an arbitrary view: the base address of each logit set
##   resolves through the padded 4-axis divmod chain on the logits shape.
## - The inner loop steps by the last-axis stride.
## - `target` is a contiguous int32 view, one index per logit set. The loss is
##   freshly allocated contiguous, one element per set.
## - CrossEntropyScalars matches the MSL `CrossEntropyParams` struct field for field, so one setBytes binds the whole scalar image.

import std/math

const kCrossEntropyMsl* = """
struct CrossEntropyParams {
  int32_t logitsOff;
  int32_t targetOff;
  int32_t logitsShape[4];
  int32_t logitsStrides[4];
  int32_t v;
  int32_t rows;
  int32_t logitStep;
};

kernel void k_crossentropy(device float* loss [[buffer(0)]],
                           device const float* logits [[buffer(1)]],
                           device const int* target [[buffer(2)]],
                           constant CrossEntropyParams& p [[buffer(3)]],
                           uint tid [[thread_position_in_grid]])
{
  int r = int(tid);
  if (r < p.rows) {
    int i0 = r * p.v;
    int c3 = i0 % p.logitsShape[3];
    int t2 = i0 / p.logitsShape[3];
    int c2 = t2 % p.logitsShape[2];
    int t1 = t2 / p.logitsShape[2];
    int c1 = t1 % p.logitsShape[1];
    int c0 = t1 / p.logitsShape[1];
    int base = p.logitsOff + (c0 * p.logitsStrides[0] + c1 * p.logitsStrides[1]) +
               c2 * p.logitsStrides[2] + c3 * p.logitsStrides[3];
    float m = logits[base];
    for (int c = 1; c < p.v; c++) {
      m = max(m, logits[base + c * p.logitStep]);
    }
    float s = 0.0;
    for (int c = 0; c < p.v; c++) {
      s += exp(logits[base + c * p.logitStep] - m);
    }
    float lse = m + log(s);
    loss[r] = lse - logits[base + target[p.targetOff + r] * p.logitStep];
  }
}
"""

type CrossEntropyScalars* = object
  ## Scalar image of one cross entropy op:
  ## - the logit and target offsets in elements
  ## - the padded logits shape and strides
  ## - v the class count and the logit set count
  ## - logitStep the class-axis stride of the logits view
  logitsOff*, targetOff*: int32
  logitsShape*, logitsStrides*: array[4, int32]
  v*, rows*: int32
  logitStep*: int32

type CrossEntropyArgs* = object
  ## Enqueue-time args object of one cross entropy op: the loss, logit and target
  ## addresses as 8-byte slots, then the scalar image.
  lossAddr*, logitsAddr*, targetAddr*: uint64
  scalars*: CrossEntropyScalars

const
  crossEntropyBufferSlots* = 3
  crossEntropyScalarsSlot* = crossEntropyBufferSlots * 8

proc packCrossEntropyArgs*(args: CrossEntropyArgs): seq[byte] =
  ## Flat byte image of one recorded cross entropy op, marshaled once at enqueue
  ## and decoded at eval.
  ##
  ## Returns:
  ## - `crossEntropyBufferSlots` 8-byte address slots
  ## - the CrossEntropyScalars image at `crossEntropyScalarsSlot`
  result = newSeq[byte](crossEntropyScalarsSlot + sizeof(CrossEntropyScalars))
  copyMem(result[0].addr, args.lossAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.logitsAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.targetAddr.unsafeAddr, 8)
  copyMem(result[crossEntropyScalarsSlot].addr, args.scalars.unsafeAddr,
      sizeof(CrossEntropyScalars))

proc decodeCrossEntropyArgs*(blob: openArray[byte]): CrossEntropyArgs =
  ## Inverse of packCrossEntropyArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(CrossEntropyArgs))

# CPU kernel body
# ------------------------------------------------------------

type CrossEntropyEnv*[T] = object
  ## Per-iteration env of the CPU cross entropy kernel, one logit set per iteration.
  loss*, logits*: ptr UncheckedArray[T]
  target*: ptr UncheckedArray[int32]
  scalars*: CrossEntropyScalars

proc crossEntropyBody*[T: SomeFloat](r: int32, env: ptr CrossEntropyEnv[T]) {.cdecl.} =
  ## CPU port of the k_crossentropy math, one logit set per iteration.
  let sc = env.scalars
  let i0 = r * sc.v
  var c: array[4, int32]
  var t = i0
  for k in countdown(3, 0):
    c[k] = t mod sc.logitsShape[k]
    t = t div sc.logitsShape[k]
  var base = sc.logitsOff
  for k in 0 .. 3:
    base += c[k] * sc.logitsStrides[k]
  var m = env.logits[][base]
  for j in 1 ..< sc.v:
    m = max(m, env.logits[][base + j * sc.logitStep])
  var s = T(0)
  for j in 0 ..< sc.v:
    s += exp(env.logits[][base + j * sc.logitStep] - m)
  let lse = m + ln(s)
  let tgt = env.target[][sc.targetOff + r]
  env.loss[][r] = lse - env.logits[][base + tgt * sc.logitStep]
