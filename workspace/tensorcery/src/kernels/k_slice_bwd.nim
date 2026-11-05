# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Range-selection backward kernel: scatter of the cotangent into the zeroed gradient of the sliced tensor.
##
## The slice forward selects count[i] elements from start[i] per axis, and its vjp
## scatters the cotangent g of the view into a fresh zeroed buffer over the parent's shape:
##   dparent[start + c] = g[c]   per view coordinate c, all axes
##
## - output element for view coordinate c: base + sum_c[d] * outStrides[d]
##   with base = sum start[d] * outStrides[d] over the parent shape strides
## - cotangent element for c: the padded 4-axis divmod chain on the cotangent shape (one thread per cotangent element)
## - The output is freshly allocated contiguous and zero-filled before the kernel runs.
##   The untouched parent elements retain their zero gradient.
##
## SliceBwdScalars matches the MSL `SliceBwdParams` struct field for field, so one setBytes binds the whole scalar image.

const kSliceBwdMsl* = """
struct SliceBwdParams {
  int32_t gOff;
  int32_t gShape[4];
  int32_t gStrides[4];
  int32_t base;
  int32_t outStrides[4];
  int32_t n;
};

kernel void k_slice_bwd(device float* dparent [[buffer(0)]],
                        device const float* g [[buffer(1)]],
                        constant SliceBwdParams& p [[buffer(2)]],
                        uint tid [[thread_position_in_grid]])
{
  int i = int(tid);
  if (i < p.n) {
    int c3 = i % p.gShape[3];
    int t2 = i / p.gShape[3];
    int c2 = t2 % p.gShape[2];
    int t1 = t2 / p.gShape[2];
    int c1 = t1 % p.gShape[1];
    int c0 = t1 / p.gShape[1];
    int gi = p.gOff + (c0 * p.gStrides[0] + c1 * p.gStrides[1]) +
             c2 * p.gStrides[2] + c3 * p.gStrides[3];
    int di = p.base + (c0 * p.outStrides[0] + c1 * p.outStrides[1]) +
             c2 * p.outStrides[2] + c3 * p.outStrides[3];
    dparent[di] = g[gi];
  }
}
"""

type SliceBwdScalars* = object
  ## Scalar image of one slice backward op:
  ##
  ## - gOff, gShape, gStrides: the cotangent offset, padded shape and strides in elements
  ## - base: the first written output element, sum start[d] * outStrides[d]
  ## - outStrides: the contiguous strides of the parent shape, padded with 0s
  ## - n: the flat element count of the cotangent
  gOff*: int32
  gShape*, gStrides*: array[4, int32]
  base*: int32
  outStrides*: array[4, int32]
  n*: int32

type SliceBwdArgs* = object
  ## Enqueue-time args object of one slice backward op: two 8-byte address slots
  ## (parent gradient, cotangent) and the scalar image.
  dparentAddr*, gAddr*: uint64
  scalars*: SliceBwdScalars

const
  sliceBwdBufferSlots* = 2
  sliceBwdScalarsSlot* = sliceBwdBufferSlots * 8

proc packSliceBwdArgs*(args: SliceBwdArgs): seq[byte] =
  ## Flat byte image of one recorded slice backward op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `sliceBwdBufferSlots` 8-byte address slots
  ## - the SliceBwdScalars image at `sliceBwdScalarsSlot`
  result = newSeq[byte](sliceBwdScalarsSlot + sizeof(SliceBwdScalars))
  copyMem(result[0].addr, args.dparentAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.gAddr.unsafeAddr, 8)
  copyMem(result[sliceBwdScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(SliceBwdScalars))

proc decodeSliceBwdArgs*(blob: openArray[byte]): SliceBwdArgs =
  ## Inverse of packSliceBwdArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(SliceBwdArgs))

# CPU kernel body
# ------------------------------------------------------------

type SliceBwdEnv*[T] = object
  ## Per-iteration env of the CPU slice backward kernel.
  dparent*, g*: ptr UncheckedArray[T]
  scalars*: SliceBwdScalars

proc sliceBwdBody*[T](idx: int32, env: ptr SliceBwdEnv[T]) {.cdecl.} =
  ## CPU port of the k_slice_bwd math: one cotangent element stores into its parent gradient position.
  let sc = env.scalars
  var c: array[4, int32]
  var t = idx
  for k in countdown(3, 0):
    c[k] = t mod sc.gShape[k]
    t = t div sc.gShape[k]
  var gi = sc.gOff
  var di = sc.base
  for k in 0 .. 3:
    gi += c[k] * sc.gStrides[k]
    di += c[k] * sc.outStrides[k]
  env.dparent[][di] = env.g[][gi]
