# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/ceramic
import workspace/tensorcery/src/tensor_datatypes
import workspace/tattleformers/src/ops_2i1o/add

export tensor_datatypes

proc runTest*[E: HwEngine](engine: E, name: string) =
  let x = newTensor(engine, @[2, 4], kBf16)
  let y = newTensor(engine, @[2, 4], kBf16)
  var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
  var yd = @[2.0'f32, 2, 2, 2, 2, 2, 2, 2]
  var xq: array[8, bfloat16]
  var yq: array[8, bfloat16]
  for i in 0 ..< 8:
    xq[i] = xd[i].to(bfloat16)
    yq[i] = yd[i].to(bfloat16)
  x.copyToDeviceFromHost(xq)
  y.copyToDeviceFromHost(yq)

  doAssert x.node.getImpl().shape().len == 2
  doAssert x.node.getImpl().shape()[0] == 2
  doAssert x.node.getImpl().shape()[1] == 4
  doAssert x.node.getImpl().strides()[0] == 4
  doAssert x.node.getImpl().strides()[1] == 1
  doAssert x.node.getImpl().dtype() == kBf16

  let z = x + y
  let parts = split(x, 2, axis = 0)
  doAssert parts.len == 2
  for p in parts:
    doAssert p.node.getImpl().shape()[0] == 1
    doAssert p.node.getImpl().shape()[1] == 4

  var small = newTensor(engine, @[1, 4], kBf16)
  doAssertRaises ValueError:
    discard x + small
  doAssertRaises ValueError:
    discard split(x, 3, axis = 0)
  doAssertRaises ValueError:
    discard newTensor(engine, @[2, 4], kFp16)
  doAssertRaises ValueError:
    x.copyToDeviceFromHost(xd)

  eval(z, parts[0])

  proc readBack(t: Tensor, n: int): seq[bfloat16] =
    result = newSeq[bfloat16](n)
    copyToHostFromDevice(result, t)

  let zGot = readBack(z, 8)
  let xyBuf = readBack(x, 8)                # split parts share x's buffer
  for i in 0 ..< 8:
    let xi = xd[i].to(bfloat16).to(float32)
    let yi = yd[i].to(bfloat16).to(float32)
    doAssert zGot[i].to(float32) == xi + yi, name & ": add " & $i
    doAssert xyBuf[i].to(float32) == xi, name & ": split source " & $i

  # mixed dtype: an f32 leaf under a bf16 add root walks and evaluates
  var xf = newTensor(engine, @[2, 4], kFp32)
  var xfd = @[1.5'f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
  xf.copyToDeviceFromHost(xfd)
  let zm = xf + y
  doAssert zm.node.getImpl().dtype() == kBf16
  var zmRoots = @[zm.node]
  zm.engine.eval(zmRoots.toSpan())
  let zmGot = readBack(zm, 8)
  for i in 0 ..< 8:
    let want = (xfd[i] + yd[i].to(bfloat16).to(float32)).to(bfloat16).to(float32)
    doAssert zmGot[i].to(float32) == want, name & ": mixed add " & $i

  # rank-1 round trip: alloc, upload, add, eval, readback
  let x1 = newTensor(engine, @[1024], kBf16)
  var x1q = newSeq[bfloat16](1024)
  for i in 0 ..< 1024:
    x1q[i] = (float32(i mod 251) - 100.0).to(bfloat16)
  x1.copyToDeviceFromHost(x1q)
  doAssert x1.node.getImpl().shape().len == 1
  doAssert x1.node.getImpl().strides()[0] == 1
  let z1 = x1 + x1
  eval(z1, x1)
  let z1Got = readBack(z1, 1024)
  for i in 0 ..< 1024:
    doAssert z1Got[i].to(float32) == 2.0 * x1q[i].to(float32), name & ": rank-1 add " & $i

  # rank-5 pool-slab round trip
  let x5 = newTensor(engine, @[2, 3, 2, 2, 3], kBf16)
  var x5q = newSeq[bfloat16](72)
  for i in 0 ..< 72:
    x5q[i] = (float32(i) * 0.5).to(bfloat16)
  x5.copyToDeviceFromHost(x5q)
  doAssert x5.node.getImpl().shape().len == 5
  doAssert x5.node.getImpl().strides()[0] == 36
  doAssert x5.node.getImpl().strides()[4] == 1
  let z5 = x5 + x5
  eval(z5, x5)
  let z5Got = readBack(z5, 72)
  for i in 0 ..< 72:
    doAssert z5Got[i].to(float32) == 2.0 * x5q[i].to(float32), name & ": rank-5 add " & $i

  # transpose view: shared buffer, one allocation, strided readback
  let xt = transpose(x)
  doAssert xt.node.getImpl().shape()[0] == 4
  doAssert xt.node.getImpl().shape()[1] == 2
  doAssert xt.node.getImpl().strides()[0] == 1
  doAssert xt.node.getImpl().strides()[1] == 4
  doAssert xt.node.getImpl().offset() == 0
  type MemX = typeof(engine.allocDevice(bfloat16, 0))
  let xImpl = x.node.getImpl().to(TensorImpl[MemX, Layout[(int, int), (int, int)]])
  let xtImpl = xt.node.getImpl().to(TensorImpl[MemX, Layout[(int, int), (int, int)]])
  doAssert not xImpl.isNil and not xtImpl.isNil
  doAssert xtImpl.data == xImpl.data, name & ": transpose view must alias the source cell"
  let xtGot = readBack(xt, 8)
  for i in 0 ..< 4:
    for j in 0 ..< 2:
      doAssert xtGot[i * 2 + j].to(float32) == xq[j * 4 + i].to(float32),
        name & ": transpose " & $i & "," & $j

  # a view over a graph node: eval materializes the source first
  let zt = transpose(z)
  eval(zt, z)
  let ztGot = readBack(zt, 8)
  for i in 0 ..< 4:
    for j in 0 ..< 2:
      doAssert ztGot[i * 2 + j].to(float32) == zGot[j * 4 + i].to(float32),
        name & ": transpose of add " & $i & "," & $j

  # narrowed window readbacks: an offset window and a strided window
  let nw = narrow(x, 0, 1, 1)
  doAssert nw.node.getImpl().shape()[0] == 1
  doAssert nw.node.getImpl().offset() == 4
  let nwGot = readBack(nw, 4)
  for i in 0 ..< 4:
    doAssert nwGot[i].to(float32) == xq[4 + i].to(float32), name & ": narrow row " & $i

  let nc = narrow(x, 1, 1, 2)
  doAssert nc.node.getImpl().shape()[1] == 2
  doAssert nc.node.getImpl().offset() == 1
  let ncGot = readBack(nc, 4)
  for r in 0 ..< 2:
    for c in 0 ..< 2:
      doAssert ncGot[r * 2 + c].to(float32) == xq[r * 4 + 1 + c].to(float32),
        name & ": narrow cols " & $r & "," & $c
  doAssertRaises ValueError:
    discard narrow(x, 0, 3, 2)

  # host uploads reach dense impls only, a view input raises
  var junk = newSeq[bfloat16](4)
  doAssertRaises ValueError:
    parts[1].copyToDeviceFromHost(junk)

  # inner-axis split: each part addresses its own columns
  let iparts = split(x, 2, axis = 1)
  doAssert iparts.len == 2
  doAssert iparts[0].node.getImpl().offset() == 0
  doAssert iparts[1].node.getImpl().offset() == 2
  let i0 = readBack(iparts[0], 4)
  let i1 = readBack(iparts[1], 4)
  for r in 0 ..< 2:
    for c in 0 ..< 2:
      doAssert i0[r * 2 + c].to(float32) == xq[r * 4 + c].to(float32),
        name & ": column split part 0 " & $r & "," & $c
      doAssert i1[r * 2 + c].to(float32) == xq[r * 4 + 2 + c].to(float32),
        name & ": column split part 1 " & $r & "," & $c

  # eval + prune: the root rebinds to a leaf. The upstream graph collects
  # under ORC, a live view and a read intermediate do not dangle
  block:
    var
      dx = newTensor(engine, @[2, 4], kBf16)
      dy = newTensor(engine, @[2, 4], kBf16)
    dx.copyToDeviceFromHost(xq)
    dy.copyToDeviceFromHost(yq)
    let
      m1 = dx + dy
      m2 = dx + dy
    var dz = m1 + m2
    let dv = narrow(m1, 1, 1, 2)
    eval(dz, m1)
    prune(dz)
    doAssert dz.node.to(AddNode).isNil, name & ": prune must replace the node by a leaf"
    let dzGot = readBack(dz, 8)
    let m1Got = readBack(m1, 8)
    let dvGot = readBack(dv, 4)
    for i in 0 ..< 8:
      let sum = xd[i].to(bfloat16).to(float32) + yd[i].to(bfloat16).to(float32)
      doAssert dzGot[i].to(float32) == 2.0 * sum, name & ": pruned root " & $i
      doAssert m1Got[i].to(float32) == sum, name & ": intermediate after prune " & $i
    for r in 0 ..< 2:
      for c in 0 ..< 2:
        doAssert dvGot[r * 2 + c].to(float32) == m1Got[r * 4 + 1 + c].to(float32),
          name & ": live view after prune " & $r & "," & $c

  # host ops: arange, full and item, no device round-trip in the op
  let ar = arange(engine, 1024)
  doAssert ar.node.getImpl().shape()[0] == 1024
  let arGot = readBack(ar, 1024)
  for i in 0 ..< 1024:
    doAssert arGot[i].to(float32) == float32(i).to(bfloat16).to(float32),
      name & ": arange " & $i

  let fu = full(engine, @[2, 3], 1.5'f32)
  doAssert fu.node.getImpl().shape()[0] == 2
  doAssert fu.node.getImpl().shape()[1] == 3
  let fuGot = readBack(fu, 6)
  for i in 0 ..< 6:
    doAssert fuGot[i].to(float32) == 1.5'f32, name & ": full " & $i

  let one = full(engine, @[1], 2.5'f32)
  doAssert item(one) == 2.5'f32, name & ": item"
  doAssertRaises ValueError:
    discard item(x)

  # metadata bridge POC: typed int32 device buffers round-trip through
  # JaggedKvHost-style staging, outside the erased Dtype face
  const metaIds = [1, 2, 3, 5, 8, 13, 21, 34]
  var metaStage = newSeq[int32](metaIds.len)
  for i in 0 ..< metaIds.len:
    metaStage[i] = int32(metaIds[i])
  let metaDev = engine.allocDevice(int32, metaIds.len)
  engine.copyToDeviceFromHost(metaDev, metaStage)
  var metaBack = newSeq[int32](metaIds.len)
  engine.copyToHostFromDevice(metaBack, metaDev)
  for i in 0 ..< metaIds.len:
    doAssert metaBack[i].int == metaIds[i], name & ": metadata bridge " & $i

  echo "tensorcery Tensor (", name, "): all checks passed"

