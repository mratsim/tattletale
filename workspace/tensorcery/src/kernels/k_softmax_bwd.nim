# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Last-axis softmax backward kernel: one thread per d-element set.
##
## With p = softmax(x) and g the cotangent of the softmax output, one set r
## with the per-set sum s = sum_c g[r, c] * p[r, c]:
##   dx[r*d + c] = p[r, c] * (g[r, c] - s)
##
## - The cotangent `g` may be an arbitrary view: the base of each d-element set
##   resolves through the padded 4-axis divmod chain on the cotangent shape.
##   The inner loop steps by the cotangent's normalized-axis stride `gStep`.
## - `p` is a freshly allocated contiguous softmax forward output, read at the linear set base r*d.
## - The output is freshly allocated contiguous, one element per input element.
##
## SoftmaxBwdScalars matches the MSL `SoftmaxBwdParams` struct field for field,
## so one setBytes binds the whole scalar image.

import std/math

const kSoftmaxBwdMsl* = """
struct SoftmaxBwdParams {
  int32_t gOff;
  int32_t gShape[4];
  int32_t gStrides[4];
  int32_t gStep;
  int32_t d;
  int32_t rows;
};

kernel void k_softmax_bwd(device float* dx [[buffer(0)]],
                          device const float* g [[buffer(1)]],
                          device const float* p [[buffer(2)]],
                          constant SoftmaxBwdParams& pb [[buffer(3)]],
                          uint tid [[thread_position_in_grid]])
{
  int r = int(tid);
  if (r < pb.rows) {
    int i0 = r * pb.d;
    int c3 = i0 % pb.gShape[3];
    int t2 = i0 / pb.gShape[3];
    int c2 = t2 % pb.gShape[2];
    int t1 = t2 / pb.gShape[2];
    int c1 = t1 % pb.gShape[1];
    int c0 = t1 / pb.gShape[1];
    int gbase = pb.gOff + (c0 * pb.gStrides[0] + c1 * pb.gStrides[1]) +
                c2 * pb.gStrides[2] + c3 * pb.gStrides[3];
    float s = 0.0;
    for (int c = 0; c < pb.d; c++) {
      s += g[gbase + c * pb.gStep] * p[i0 + c];
    }
    for (int c = 0; c < pb.d; c++) {
      dx[i0 + c] = p[i0 + c] * (g[gbase + c * pb.gStep] - s);
    }
  }
}
"""

type SoftmaxBwdScalars* = object
  ## Scalar image of one softmax backward op:
  ##
  ## - gOff: the cotangent offset in elements
  ## - gShape, gStrides: the padded cotangent shape and strides
  ## - gStep: the cotangent's normalized-axis stride
  ## - d: the normalized axis size and the d-element set count
  gOff*: int32
  gShape*, gStrides*: array[4, int32]
  gStep*, d*, rows*: int32

type SoftmaxBwdArgs* = object
  ## Enqueue-time args object of one softmax backward op: three 8-byte address
  ## slots (output, cotangent, softmax output) and the scalar image.
  dxAddr*, gAddr*, pAddr*: uint64
  scalars*: SoftmaxBwdScalars

const
  softmaxBwdBufferSlots* = 3
  softmaxBwdScalarsSlot* = softmaxBwdBufferSlots * 8

proc packSoftmaxBwdArgs*(args: SoftmaxBwdArgs): seq[byte] =
  ## Flat byte image of one recorded softmax backward op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `softmaxBwdBufferSlots` 8-byte address slots
  ## - the SoftmaxBwdScalars image at `softmaxBwdScalarsSlot`
  result = newSeq[byte](softmaxBwdScalarsSlot + sizeof(SoftmaxBwdScalars))
  copyMem(result[0].addr, args.dxAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.gAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.pAddr.unsafeAddr, 8)
  copyMem(result[softmaxBwdScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(SoftmaxBwdScalars))

proc decodeSoftmaxBwdArgs*(blob: openArray[byte]): SoftmaxBwdArgs =
  ## Inverse of packSoftmaxBwdArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(SoftmaxBwdArgs))

# CPU kernel body
# ------------------------------------------------------------

type SoftmaxBwdEnv*[T] = object
  ## Per-iteration env of the CPU softmax backward kernel, one d-element set per iteration.
  dx*, g*, p*: ptr UncheckedArray[T]
  scalars*: SoftmaxBwdScalars

proc softmaxBwdBody*[T: SomeFloat](r: int32, env: ptr SoftmaxBwdEnv[T]) {.cdecl.} =
  ## CPU port of the k_softmax_bwd math, one set of d elements per iteration.
  let sc = env.scalars
  let i0 = r * sc.d
  var c: array[4, int32]
  var t = i0
  for k in countdown(3, 0):
    c[k] = t mod sc.gShape[k]
    t = t div sc.gShape[k]
  var gbase = sc.gOff
  for k in 0 .. 3:
    gbase += c[k] * sc.gStrides[k]
  var s = T(0)
  for j in 0 ..< sc.d:
    s += env.g[][gbase + j * sc.gStep] * env.p[][i0 + j]
  for j in 0 ..< sc.d:
    env.dx[][i0 + j] = env.p[][i0 + j] * (env.g[][gbase + j * sc.gStep] - s)
