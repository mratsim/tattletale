# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Cross entropy backward kernel: one thread per logit set.
##
## With lse = logsumexp of the set's v logits (as k_crossentropy computes it),
## the cotangent g of the loss and target[r] the named class of set r:
##   dlogits[r*v + c] = g[r] * (exp(logits[r, c] - lse) - (c == target[r] ? 1 : 0))
##
## - The cotangent `g` may be an arbitrary view: the loss element of set r
##   resolves through the padded 4-axis divmod chain on the cotangent shape.
## - The logits may be an arbitrary view: the base of each logit set resolves
##   through the padded divmod chain on the logits shape, and the inner loop
##   steps by the logits' class-axis stride `logitStep`.
## - `target` is a contiguous int32 view, one index per logit set.
## - The output is freshly allocated contiguous, one element per logit.
##
## CrossEntropyBwdScalars matches the MSL `CrossEntropyBwdParams` struct field for field, so one setBytes binds the whole scalar image.

import std/math

const kCrossEntropyBwdMsl* = """
struct CrossEntropyBwdParams {
  int32_t gOff;
  int32_t gShape[4];
  int32_t gStrides[4];
  int32_t logitsOff;
  int32_t logitsShape[4];
  int32_t logitsStrides[4];
  int32_t logitStep;
  int32_t targetOff;
  int32_t v;
  int32_t rows;
};

kernel void k_crossentropy_bwd(device float* dlogits [[buffer(0)]],
                               device const float* g [[buffer(1)]],
                               device const float* logits [[buffer(2)]],
                               device const int* target [[buffer(3)]],
                               constant CrossEntropyBwdParams& p [[buffer(4)]],
                               uint tid [[thread_position_in_grid]])
{
  int r = int(tid);
  if (r < p.rows) {
    int c3 = r % p.gShape[3];
    int t2 = r / p.gShape[3];
    int c2 = t2 % p.gShape[2];
    int t1 = t2 / p.gShape[2];
    int c1 = t1 % p.gShape[1];
    int c0 = t1 / p.gShape[1];
    int gi = p.gOff + (c0 * p.gStrides[0] + c1 * p.gStrides[1]) +
             c2 * p.gStrides[2] + c3 * p.gStrides[3];
    int i0 = r * p.v;
    int lc3 = i0 % p.logitsShape[3];
    int lt2 = i0 / p.logitsShape[3];
    int lc2 = lt2 % p.logitsShape[2];
    int lt1 = lt2 / p.logitsShape[2];
    int lc1 = lt1 % p.logitsShape[1];
    int lc0 = lt1 / p.logitsShape[1];
    int lbase = p.logitsOff + (lc0 * p.logitsStrides[0] + lc1 * p.logitsStrides[1]) +
                lc2 * p.logitsStrides[2] + lc3 * p.logitsStrides[3];
    float m = logits[lbase];
    for (int c = 1; c < p.v; c++) {
      m = max(m, logits[lbase + c * p.logitStep]);
    }
    float s = 0.0;
    for (int c = 0; c < p.v; c++) {
      s += exp(logits[lbase + c * p.logitStep] - m);
    }
    float lse = m + log(s);
    float gv = g[gi];
    int tgt = target[p.targetOff + r];
    for (int c = 0; c < p.v; c++) {
      float d = exp(logits[lbase + c * p.logitStep] - lse) - (c == tgt ? 1.0 : 0.0);
      dlogits[i0 + c] = gv * d;
    }
  }
}
"""

type CrossEntropyBwdScalars* = object
  ## Scalar image of one cross entropy backward op:
  ##
  ## - gOff, logitsOff, targetOff: the cotangent, logits and target offsets in elements
  ## - gShape, gStrides: the padded cotangent shape and strides
  ## - logitsShape, logitsStrides, logitStep: the padded logits shape, strides and class-axis stride
  ## - v: the class count and the logit set count
  gOff*: int32
  gShape*, gStrides*: array[4, int32]
  logitsOff*: int32
  logitsShape*, logitsStrides*: array[4, int32]
  logitStep*, targetOff*: int32
  v*, rows*: int32

type CrossEntropyBwdArgs* = object
  ## Enqueue-time args object of one cross entropy backward op: four 8-byte address
  ## slots (output, cotangent, logits, target) and the scalar image.
  dlogitsAddr*, gAddr*, logitsAddr*, targetAddr*: uint64
  scalars*: CrossEntropyBwdScalars

const
  crossEntropyBwdBufferSlots* = 4
  crossEntropyBwdScalarsSlot* = crossEntropyBwdBufferSlots * 8

proc packCrossEntropyBwdArgs*(args: CrossEntropyBwdArgs): seq[byte] =
  ## Flat byte image of one recorded cross entropy backward op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `crossEntropyBwdBufferSlots` 8-byte address slots
  ## - the CrossEntropyBwdScalars image at `crossEntropyBwdScalarsSlot`
  result = newSeq[byte](crossEntropyBwdScalarsSlot + sizeof(CrossEntropyBwdScalars))
  copyMem(result[0].addr, args.dlogitsAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.gAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.logitsAddr.unsafeAddr, 8)
  copyMem(result[24].addr, args.targetAddr.unsafeAddr, 8)
  copyMem(result[crossEntropyBwdScalarsSlot].addr, args.scalars.unsafeAddr,
      sizeof(CrossEntropyBwdScalars))

proc decodeCrossEntropyBwdArgs*(blob: openArray[byte]): CrossEntropyBwdArgs =
  ## Inverse of packCrossEntropyBwdArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(CrossEntropyBwdArgs))

# CPU kernel body
# ------------------------------------------------------------

type CrossEntropyBwdEnv*[T] = object
  ## Per-iteration env of the CPU cross entropy backward kernel, one logit set per iteration.
  dlogits*, g*, logits*: ptr UncheckedArray[T]
  target*: ptr UncheckedArray[int32]
  scalars*: CrossEntropyBwdScalars

proc crossEntropyBwdBody*[T: SomeFloat](r: int32, env: ptr CrossEntropyBwdEnv[T]) {.cdecl.} =
  ## CPU port of the k_crossentropy_bwd math, one logit set per iteration.
  let sc = env.scalars
  var c: array[4, int32]
  var t = r
  for k in countdown(3, 0):
    c[k] = t mod sc.gShape[k]
    t = t div sc.gShape[k]
  var gi = sc.gOff
  for k in 0 .. 3:
    gi += c[k] * sc.gStrides[k]
  let i0 = r * sc.v
  var lbase = sc.logitsOff
  var lc: array[4, int32]
  var lt = i0
  for k in countdown(3, 0):
    lc[k] = lt mod sc.logitsShape[k]
    lt = lt div sc.logitsShape[k]
  for k in 0 .. 3:
    lbase += lc[k] * sc.logitsStrides[k]
  var m = env.logits[][lbase]
  for j in 1 ..< sc.v:
    m = max(m, env.logits[][lbase + j * sc.logitStep])
  var s = T(0)
  for j in 0 ..< sc.v:
    s += exp(env.logits[][lbase + j * sc.logitStep] - m)
  let lse = m + ln(s)
  let gv = env.g[][gi]
  let tgt = env.target[][sc.targetOff + r]
  for j in 0 ..< sc.v:
    let d = exp(env.logits[][lbase + j * sc.logitStep] - lse) -
        (if j == tgt: T(1) else: T(0))
    env.dlogits[][i0 + j] = gv * d
