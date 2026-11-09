# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/tensorcery/src/tensor_datatypes

export tensor_datatypes

proc runTest*(engine: AnyEngine, name: string) =
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
  let (q, r) = divmod(x, y)
  let parts = chunk(x, 0, 2)
  doAssert parts.len == 2
  for p in parts:
    doAssert p.node.getImpl().shape()[0] == 1
    doAssert p.node.getImpl().shape()[1] == 4

  var small = newTensor(engine, @[1, 4], kBf16)
  doAssertRaises ValueError:
    discard x + small
  doAssertRaises ValueError:
    discard chunk(x, 0, 3)
  doAssertRaises ValueError:
    discard newTensor(engine, @[2, 4], kFp16)
  doAssertRaises ValueError:
    x.copyToDeviceFromHost(xd)

  eval(z, q, r, parts[0])

  proc readBack(t: Tensor, n: int): seq[bfloat16] =
    result = newSeq[bfloat16](n)
    copyToHostFromDevice(result, t)

  let zGot = readBack(z, 8)
  let qGot = readBack(q, 8)
  let rGot = readBack(r, 8)
  let xyBuf = readBack(x, 8)                # chunks share x's buffer
  for i in 0 ..< 8:
    let xi = xd[i].to(bfloat16).to(float32)
    let yi = yd[i].to(bfloat16).to(float32)
    doAssert zGot[i].to(float32) == xi + yi, name & ": add " & $i
    let d = xi / yi
    var qi = float32(int32(d))
    if qi > d:
      qi -= 1.0
    doAssert qGot[i].to(float32) == qi, name & ": div q " & $i
    doAssert rGot[i].to(float32) == xi - qi * yi, name & ": div r " & $i
    doAssert xyBuf[i].to(float32) == xi, name & ": chunk source " & $i

  echo "tensorcery Tensor (", name, "): all checks passed"

