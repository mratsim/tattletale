# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/importutils
import workspace/tensorcery/src/tensorcery
import workspace/tensorcery/src/tensors_internals
import workspace/tensorcery/src/tensors_globals
import workspace/tensorcery/src/compute_graph/eval
export eval.eval

export tensorcery
# erased layer imports for the white-box tests: iface vtable mixins
# resolve at the outermost instantiation scope
export tensors_internals

proc runTest*(run: static Runtime, name: string) =
  privateAccess(Tensor)
  let engine = gEngine(run)
  var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
  var yd = @[2.0'f32, 2, 2, 2, 2, 2, 2, 2]
  var xq: array[8, bfloat16]
  var yq: array[8, bfloat16]
  for i in 0 ..< 8:
    xq[i] = xd[i].to(bfloat16)
    yq[i] = yd[i].to(bfloat16)
  let x = toTensor(xq, @[2, 4])
  let y = toTensor(yq, @[2, 4])

  doAssert x.node.getImpl().shape().len == 2
  doAssert x.node.getImpl().shape()[0] == 2
  doAssert x.node.getImpl().shape()[1] == 4
  doAssert x.node.getImpl().strides()[0] == 4
  doAssert x.node.getImpl().strides()[1] == 1
  doAssert x.node.getImpl().dtype() == kBf16

  let z = x + y
  let parts = split(x, 0, 2)
  doAssert parts.len == 2
  for p in parts:
    doAssert p.node.getImpl().shape()[0] == 1
    doAssert p.node.getImpl().shape()[1] == 4

  let small = toTensor(yq[0 ..< 4], @[1, 4])
  # face checks are contracts, they raise AssertionDefect
  doAssertRaises AssertionDefect:
    discard x + small
  doAssertRaises AssertionDefect:
    discard split(x, 0, 3)
  doAssertRaises AssertionDefect:
    discard toTensor(xq[0 ..< 7], @[2, 4])

  eval(z, parts[0])

  var zGot = newSeq[bfloat16](8)
  z.toHost(zGot)
  var xyBuf = newSeq[bfloat16](8)
  x.toHost(xyBuf)                           # split parts share x's buffer
  var p0Got = newSeq[bfloat16](4)
  parts[0].toHost(p0Got)
  var p1Got = newSeq[bfloat16](4)
  parts[1].toHost(p1Got)
  # inner-axis split makes strided views, strides inherit the source
  let cols = split(x, 1, 2)
  let w = cols[0] + cols[1]
  eval(w)
  var wGot = newSeq[bfloat16](4)
  w.toHost(wGot)
  for i in 0 ..< 8:
    let xi = xd[i].to(bfloat16).to(float32)
    let yi = yd[i].to(bfloat16).to(float32)
    doAssert zGot[i].to(float32) == xi + yi, name & ": add " & $i
    doAssert xyBuf[i].to(float32) == xi, name & ": split source " & $i
  for i in 0 ..< 4:
    doAssert p0Got[i].to(float32) == xd[i].to(bfloat16).to(float32), name & ": part0 " & $i
    doAssert p1Got[i].to(float32) == xd[4 + i].to(bfloat16).to(float32), name & ": part1 " & $i
  for r in 0 ..< 2:
    for c in 0 ..< 2:
      let e0 = xd[r * 4 + c].to(bfloat16).to(float32)
      let e1 = xd[r * 4 + c + 2].to(bfloat16).to(float32)
      doAssert wGot[r * 2 + c].to(float32) == e0 + e1, name & ": colsplit " & $r & "," & $c

  echo "tensorcery Tensor (", name, "): all checks passed"

