# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Last-axis softmax kernel: one thread per set of d elements with max subtraction
## for numerical stability.
##
## For element i, r = i div d and c = i mod d:
##   out[i]  = exp(x[r, c] - m) / sum_c exp(x[r, c] - m)
##   with m = max_c x[r, c]
##
## - The input may be an arbitrary view: the base address of each d-element set
##   resolves through the padded 4-axis divmod chain on the input shape.
## - The inner loop steps by the last-axis stride.
## - The output is freshly allocated contiguous.
## - SoftmaxScalars matches the MSL `SoftmaxParams` struct field for field, so one setBytes binds the whole scalar image.

import std/math

const kSoftmaxMsl* = """
struct SoftmaxParams {
  int32_t xOff;
  int32_t xShape[4];
  int32_t xs[4];
  int32_t d;
  int32_t rows;
  int32_t xStep;
};

kernel void k_softmax(device float* output [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      constant SoftmaxParams& p [[buffer(2)]],
                      uint tid [[thread_position_in_grid]])
{
  int r = int(tid);
  if (r < p.rows) {
    int i0 = r * p.d;
    int c3 = i0 % p.xShape[3];
    int t2 = i0 / p.xShape[3];
    int c2 = t2 % p.xShape[2];
    int t1 = t2 / p.xShape[2];
    int c1 = t1 % p.xShape[1];
    int c0 = t1 / p.xShape[1];
    int base = p.xOff + (c0 * p.xs[0] + c1 * p.xs[1]) + c2 * p.xs[2] + c3 * p.xs[3];
    float m = x[base];
    for (int c = 1; c < p.d; c++) {
      m = max(m, x[base + c * p.xStep]);
    }
    float s = 0.0;
    for (int c = 0; c < p.d; c++) {
      float e = exp(x[base + c * p.xStep] - m);
      output[i0 + c] = e;
      s += e;
    }
    for (int c = 0; c < p.d; c++) {
      output[i0 + c] /= s;
    }
  }
}
"""

type SoftmaxScalars* = object
  ## Scalar image of one softmax op:
  ## - the input offset in elements
  ## - the padded input shape and strides
  ## - d the normalized axis size and the d-element set count
  ## - xStep the normalized-axis stride of the input view
  xOff*: int32
  xShape*, xs*: array[4, int32]
  d*, rows*: int32
  xStep*: int32

type SoftmaxArgs* = object
  ## Enqueue-time args object of one softmax op: the output and input addresses
  ## as 8-byte slots, then the scalar image.
  outAddr*, xAddr*: uint64
  scalars*: SoftmaxScalars

const
  softmaxBufferSlots* = 2
  softmaxScalarsSlot* = softmaxBufferSlots * 8

proc packSoftmaxArgs*(args: SoftmaxArgs): seq[byte] =
  ## Flat byte image of one recorded softmax op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `softmaxBufferSlots` 8-byte address slots
  ## - the SoftmaxScalars image at `softmaxScalarsSlot`
  result = newSeq[byte](softmaxScalarsSlot + sizeof(SoftmaxScalars))
  copyMem(result[0].addr, args.outAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.xAddr.unsafeAddr, 8)
  copyMem(result[softmaxScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(SoftmaxScalars))

proc decodeSoftmaxArgs*(blob: openArray[byte]): SoftmaxArgs =
  ## Inverse of packSoftmaxArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(SoftmaxArgs))

# CPU kernel body
# ------------------------------------------------------------

type SoftmaxEnv*[T] = object
  ## Per-iteration env of the CPU softmax kernel, one d-element set per iteration.
  outp*, x*: ptr UncheckedArray[T]
  scalars*: SoftmaxScalars

proc softmaxBody*[T: SomeFloat](r: int32, env: ptr SoftmaxEnv[T]) {.cdecl.} =
  ## CPU port of the k_softmax math, one set of d elements per iteration.
  let sc = env.scalars
  let i0 = r * sc.d
  var c: array[4, int32]
  var t = i0
  for k in countdown(3, 0):
    c[k] = t mod sc.xShape[k]
    t = t div sc.xShape[k]
  var base = sc.xOff
  for k in 0 .. 3:
    base += c[k] * sc.xs[k]
  var m = env.x[][base]
  for j in 1 ..< sc.d:
    m = max(m, env.x[][base + j * sc.xStep])
  var s = T(0)
  for j in 0 ..< sc.d:
    let e = exp(env.x[][base + j * sc.xStep] - m)
    env.outp[][i0 + j] = e
    s += e
  for j in 0 ..< sc.d:
    env.outp[][i0 + j] /= s
