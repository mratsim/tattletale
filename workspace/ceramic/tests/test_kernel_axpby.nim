## Test: kernel_axpby — axpby epilogue (GPU path)
##
## Tests axpby(alpha, X, beta, Y) = Y = α·X + β·Y
## against manual computation.

import ../src/int_tuples
import ../src/layouts
import ../src/tensors
import ../src/ptr_arithmetic
import ../src/kernel_axpby_gpu

{.experimental: "callOperator".}

template test(label: string; body: untyped) =
  block:
    body
  echo "  [OK] ", label

# ═══════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════

proc runAxpbyTests =
  test "axpby identity: α=1, β=1, Y=0":
    var xBuf = newSeq[float32](4)
    var yBuf = newSeq[float32](4)
    for i in 0 ..< 4: xBuf[i] = float32(i + 1)
    for i in 0 ..< 4: yBuf[i] = 0.0'f32

    let X = make_view(xBuf +% 0, make_layout((4,)))
    var Y = make_view(yBuf +% 0, make_layout((4,)))
    axpby(1.0'f32, X, 1.0'f32, Y)

    doAssert Y[0] == 1.0'f32
    doAssert Y[1] == 2.0'f32
    doAssert Y[2] == 3.0'f32
    doAssert Y[3] == 4.0'f32

  test "axpby scale only: α=2, β=1":
    var xBuf = newSeq[float32](3)
    var yBuf = newSeq[float32](3)
    for i in 0 ..< 3: xBuf[i] = float32(i + 1)
    for i in 0 ..< 3: yBuf[i] = 0.0'f32

    let X = make_view(xBuf +% 0, make_layout((3,)))
    var Y = make_view(yBuf +% 0, make_layout((3,)))
    axpby(2.0'f32, X, 1.0'f32, Y)

    doAssert Y[0] == 2.0'f32
    doAssert Y[1] == 4.0'f32
    doAssert Y[2] == 6.0'f32

  test "axpby α=1, β=2 with pre-filled Y":
    var xBuf = newSeq[float32](3)
    var yBuf = newSeq[float32](3)
    for i in 0 ..< 3: xBuf[i] = float32(i + 1)    # [1, 2, 3]
    for i in 0 ..< 3: yBuf[i] = 1.0'f32            # [1, 1, 1]

    let X = make_view(xBuf +% 0, make_layout((3,)))
    var Y = make_view(yBuf +% 0, make_layout((3,)))
    axpby(1.0'f32, X, 2.0'f32, Y)
    # Y = 1*X + 2*Y = [1+2, 2+2, 3+2] = [3, 4, 5]

    doAssert Y[0] == 3.0'f32
    doAssert Y[1] == 4.0'f32
    doAssert Y[2] == 5.0'f32

  test "axpby fractional α=0.5, β=0.5":
    var xBuf = newSeq[float32](3)
    var yBuf = newSeq[float32](3)
    for i in 0 ..< 3: xBuf[i] = float32((i + 1) * 2)  # [2, 4, 6]
    for i in 0 ..< 3: yBuf[i] = 2.0'f32                # [2, 2, 2]

    let X = make_view(xBuf +% 0, make_layout((3,)))
    var Y = make_view(yBuf +% 0, make_layout((3,)))
    axpby(0.5'f32, X, 0.5'f32, Y)
    # Y = 0.5*X + 0.5*Y = [1+1, 2+1, 3+1] = [2, 3, 4]

    doAssert Y[0] == 2.0'f32
    doAssert Y[1] == 3.0'f32
    doAssert Y[2] == 4.0'f32

  test "axpby β=0 (ignore Y)" :
    var xBuf = newSeq[float32](3)
    var yBuf = newSeq[float32](3)
    for i in 0 ..< 3: xBuf[i] = float32(i + 1)       # [1, 2, 3]
    for i in 0 ..< 3: yBuf[i] = 99.0'f32             # [99, 99, 99]

    let X = make_view(xBuf +% 0, make_layout((3,)))
    var Y = make_view(yBuf +% 0, make_layout((3,)))
    axpby(2.0'f32, X, 0.0'f32, Y)
    # Y = 2*X + 0*Y = [2, 4, 6]

    doAssert Y[0] == 2.0'f32
    doAssert Y[1] == 4.0'f32
    doAssert Y[2] == 6.0'f32

  test "axpby 2D tensor":
    var xBuf = newSeq[float32](6)
    var yBuf = newSeq[float32](6)
    for i in 0 ..< 6: xBuf[i] = 1.0'f32
    for i in 0 ..< 6: yBuf[i] = 2.0'f32

    let X = make_view(xBuf +% 0, make_layout((2, 3), (1, 2)))
    var Y = make_view(yBuf +% 0, make_layout((2, 3), (1, 2)))
    axpby(3.0'f32, X, 1.0'f32, Y)
    # Y = 3*1 + 1*2 = 5

    for m in 0 ..< 2:
      for n in 0 ..< 3:
        doAssert Y[m, n] == 5.0'f32

when isMainModule:
  runAxpbyTests()
  echo "OK: all axpby tests passed"
