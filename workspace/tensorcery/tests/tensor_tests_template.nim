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

  echo "tensorcery Tensor (", name, "): all checks passed"

