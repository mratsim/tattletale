# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import ../tensorcery_testutils

export tensorcery_testutils

from std/importutils import privateAccess
import workspace/tensorcery/src/compute_graph/ops_metadata/ops_views {.all.}

proc runViewsTest*(run: static Runtime, name: string) =
  privateAccess(Tensor)

  # Reshape of an op output is a rank-changing view binding at eval, same element order
  block:
    ## Nim: reshape(x + x, [2, 2, 2])
    ## MLX: mx.reshape(x + x, (2, 2, 2))
    ## PyTorch: (x + x).reshape(2, 2, 2)
    var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
    var x = xd.toTensor([2, 4], run, bfloat16)
    let y = x + x
    let r = reshape(y, [2, 2, 2])
    doAssert r.shape()[0] == 2 and r.shape()[1] == 2 and r.shape()[2] == 2
    eval(r)
    var expected: array[8, bfloat16]
    for i in 0 ..< 8:
      expected[i] = (2.0'f32 * xd[i]).to(bfloat16)
    doAssert r.allClose(expected.toTensor([2, 2, 2], run, bfloat16)), name & ": reshape"

  # Transpose with no axes reverses every axis. With axes, result axis i reads source axis axes[i]
  block:
    ## Nim: transpose(x) / transpose(x, [2, 0, 1])
    ## MLX: mx.transpose(x) / mx.transpose(x, (2, 0, 1))
    ## PyTorch: x.T (2-D) / x.permute(2, 0, 1)
    var xd = @[1.0'f32, 2, 3, 4, 5, 6]
    var x = xd.toTensor([2, 3], run, bfloat16)
    let t = transpose(x)
    doAssert t.shape()[0] == 3 and t.shape()[1] == 2
    doAssert t.allClose([xd[0], xd[3], xd[1], xd[4], xd[2], xd[5]].toTensor([3, 2], run, bfloat16)),
        name & ": transpose"
    var x3d = xd.toTensor([1, 2, 3], run, bfloat16)
    let p = transpose(x3d, [2, 0, 1])
    doAssert p.shape()[0] == 3 and p.shape()[1] == 1 and p.shape()[2] == 2
    doAssert p.allClose([xd[0], xd[3], xd[1], xd[4], xd[2], xd[5]].toTensor([3, 1, 2], run, bfloat16)),
        name & ": transpose axes"

  # Squeeze removes a size-1 axis, rank changes survive eval on an op output
  block:
    ## Nim: squeeze(x + x, 1)
    ## MLX: mx.squeeze(x + x, 1)
    ## PyTorch: (x + x).squeeze(1)
    var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
    var x = xd.toTensor([2, 1, 4], run, bfloat16)
    let y = x + x
    let sq = squeeze(y, 1)
    doAssert sq.shape()[0] == 2 and sq.shape()[1] == 4
    eval(sq)
    var expected: array[8, bfloat16]
    for i in 0 ..< 8:
      expected[i] = (2.0'f32 * xd[i]).to(bfloat16)
    doAssert sq.allClose(expected.toTensor([2, 4], run, bfloat16)), name & ": squeeze"

  # Expand_dims inserts a size-1 axis with the canonical dense stride,
  # so a later reshape stays order-preserving
  block:
    ## Nim: expand_dims(x, 1) / reshape(expand_dims(x, 1), [2, 4])
    ## MLX: mx.expand_dims(x, 1) / mx.reshape(mx.expand_dims(x, 1), (2, 4))
    ## PyTorch: x.unsqueeze(1) / x.unsqueeze(1).reshape(2, 4)
    var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
    var x = xd.toTensor([2, 4], run, bfloat16)
    let e = expand_dims(x, 1)
    doAssert e.shape()[0] == 2 and e.shape()[1] == 1 and e.shape()[2] == 4
    doAssert e.allClose(xd.toTensor([2, 1, 4], run, bfloat16)), name & ": expand_dims"
    doAssert reshape(e, [2, 4]).allClose(xd.toTensor([2, 4], run, bfloat16)),
        name & ": reshape after expand_dims"

  # Broadcast of a narrow slice evals and reads stride-0 axes on device:
  # slice (2, 2) at column offset 1 stretches a leading axis of 3, then doubles
  block:
    ## Nim: broadcast_to(narrow(x + x, 1, 1, 2), [3, 2, 2]) + broadcast_to(...)
    ## MLX: mx.broadcast_to((x + x)[:, 1:3], (3, 2, 2))
    ## PyTorch: (x + x)[:, 1:3].expand(3, 2, 2)
    var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
    var x = xd.toTensor([2, 4], run, bfloat16)
    let y = x + x
    let slice = narrow(y, 1, 1, 2)
    doAssert slice.shape()[0] == 2 and slice.shape()[1] == 2
    doAssert slice.strides()[0] == 4 and slice.strides()[1] == 1
    let b = broadcast_to(slice, [3, 2, 2])
    let w = b + b
    eval(w)
    var expected: array[12, bfloat16]
    for i0 in 0 ..< 3:
      for i in 0 ..< 2:
        for j in 0 ..< 2:
          # slice reads 2*xd and the add doubles it again
          expected[i0 * 4 + i * 2 + j] = (4.0'f32 * xd[i * 4 + 1 + j]).to(bfloat16)
    doAssert w.allClose(expected.toTensor([3, 2, 2], run, bfloat16)), name & ": broadcast_to narrow"

  # A negative-stride view is legal: (4,)/[-1] at offset 3 stays inside a 4-element root
  block:
    ## Nim: view(x, [4], [-1], 3), a reversed view of the 4-element root
    ## MLX: x[::-1], axis reversed
    ## PyTorch: x.flip(0), flipped along axis 0
    var xd = @[10.0'f32, 20, 30, 40]
    var x = xd.toTensor([4], run, bfloat16)
    let vNode = x.engine.tn_view(x.node, [4], [-1], 3)
    let v = Tensor(engine: x.engine, node: vNode)
    var expected: array[4, bfloat16]
    for i in 0 ..< 4:
      expected[i] = xd[3 - i].to(bfloat16)
    doAssert v.allClose(expected.toTensor([4], run, bfloat16)), name & ": negative stride"

  # (2,)/[-3] at offset 2 reaches element -1, below the root buffer: refused
  block:
    var x = [1.0'f32, 2, 3, 4].toTensor([4], run, bfloat16)
    doAssertRaises AssertionDefect:
      discard x.engine.tn_view(x.node, [2], [-3], 2)

  # Column-major (2,2)/(1,2) has cosize == size == 4, yet a reshape to (4,)
  # scrambles element order. The source strides are not canonical dense strides
  # and reshape refuses.
  block:
    var xd = @[1.0'f32, 2, 3, 4]
    var x = xd.toTensor([4], run, bfloat16)
    let cmNode = x.engine.tn_view(x.node, [2, 2], [1, 2], 0)
    let cm = Tensor(engine: x.engine, node: cmNode)
    doAssertRaises AssertionDefect:
      discard reshape(cm, [4])

  # Reshape refusals: a non-dense source, and the three invalid -1 forms
  block:
    # narrow (1, seq, hidden) axis 0 to width 1: the axis keeps stride
    # seq*hidden, not the canonical dense stride of (1, 1, hidden)
    var xd: seq[float32]
    for i in 0 ..< 12:
      xd.add i.float32
    var x = xd.toTensor([1, 3, 4], run, bfloat16)
    let s = narrow(x, 0, 0, 1)
    doAssertRaises AssertionDefect:
      discard reshape(s, [4])
    var x2 = xd.toTensor([2, 6], run, bfloat16)
    doAssertRaises AssertionDefect:
      discard reshape(x2, [-1, -1])
    doAssertRaises AssertionDefect:
      discard reshape(x2, [0, -1])
    var x3 = xd[0 ..< 6].toTensor([2, 3], run, bfloat16)
    doAssertRaises AssertionDefect:
      discard reshape(x3, [4, -1])

  # Narrow never trusts its source: an out-of-window view built on the raw
  # machinery boundary is refused against the root buffer
  block:
    var x = [1.0'f32, 2, 3, 4].toTensor([4], run, bfloat16)
    let bad = x.engine.newViewNode(x.node, [4], [-1], 0, "bad")
    doAssertRaises AssertionDefect:
      discard x.engine.tn_narrow(bad, 0, 1, 2)

  # Transpose axes must be a bijection of the source axes
  block:
    var x = [1.0'f32, 2, 3, 4, 5, 6, 7, 8].toTensor([2, 2, 2], run, bfloat16)
    doAssertRaises AssertionDefect:
      discard transpose(x, [0, 0, 2])

  # Broadcast refusals: rank decrease, mismatched trailing size, negative dims
  block:
    var x = [1.0'f32, 2, 3, 4, 5, 6].toTensor([2, 3], run, bfloat16)
    doAssertRaises AssertionDefect:
      discard broadcast_to(x, [3])
    doAssertRaises AssertionDefect:
      discard broadcast_to(x, [2, 4])
    doAssertRaises AssertionDefect:
      discard broadcast_to(x, [-3, 3])

  # Squeeze refusals: the axis must be size 1 and the rank stays above 0.
  # Expand_dims refusal: the axis must be a valid insertion point
  block:
    var x = [1.0'f32, 2, 3, 4, 5, 6].toTensor([2, 3], run, bfloat16)
    doAssertRaises AssertionDefect:
      discard squeeze(x, 1)
    var x1 = [1.0'f32, 2, 3].toTensor([3], run, bfloat16)
    doAssertRaises AssertionDefect:
      discard squeeze(x1, 0)
    doAssertRaises AssertionDefect:
      discard expand_dims(x, 3)

  echo "tensorcery views (", name, "): all checks passed"
