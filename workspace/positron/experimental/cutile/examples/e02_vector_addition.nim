## cutile/examples/e02_vector_addition.nim
##
## Vector addition: z[i] = x[i] + y[i]
##
##  1. Pure Nim reference implementation (CPU)
##  2. TileIR bytecode builder emitting the same computation
##  3. Structural verification: bytecode ops match the reference pattern
##  4. GPU execution (when tileiras/CUDA 13.1+ JIT is available)

import
  std/[os, strutils, math],
  ../bytecode,
  ../dsl,
  ../cuda_driver,
  ../compiler

# ############################################################
# 1. Reference implementation (pure Nim, CPU)
# ############################################################

proc refVecAdd*(x, y: openArray[float32]): seq[float32] =
  doAssert x.len == y.len
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len:
    result[i] = x[i] + y[i]

proc refVecAddAllClose*(x, y, z: openArray[float32],
                        rtol = 1e-5'f32): bool =
  let expected = refVecAdd(x, y)
  for i in 0 ..< z.len:
    let diff = abs(expected[i] - z[i])
    let maxAbs = max(abs(expected[i]), abs(z[i]))
    if maxAbs > 0 and diff / maxAbs > rtol: return false
    elif maxAbs == 0 and diff > rtol: return false
  return true

# ############################################################
# 2. TileIR kernel builder
# ############################################################

proc buildVecAddKernel*(tileSize: int64): BytecodeModule =
  let ptrScalar = TileType(shape: @[], elemType: ElemPointer)
  let tileSizeT = TileType(shape: @[tileSize], elemType: ElemF32)
  let tileSizePtr = TileType(shape: @[tileSize], elemType: ElemPointer)
  let tile1Ptr = TileType(shape: @[1], elemType: ElemPointer)

  let kb = newKernel("vec_add_kernel", @[ptrScalar, ptrScalar, ptrScalar], @[])
  let iota = kb.iota(@[tileSize], ElemI32)

  let xPtr1 = kb.reshape(0, tile1Ptr)
  let xPtrTile = kb.broadcast(xPtr1, tileSizePtr)
  let xPtrs = kb.offset(xPtrTile, iota, tileSizePtr)
  let (xData, _) = kb.loadPtrTko(xPtrs, tileSizeT)

  let yPtr1 = kb.reshape(1, tile1Ptr)
  let yPtrTile = kb.broadcast(yPtr1, tileSizePtr)
  let yPtrs = kb.offset(yPtrTile, iota, tileSizePtr)
  let (yData, _) = kb.loadPtrTko(yPtrs, tileSizeT)

  let zData = kb.addF(xData, yData, tileSizeT)

  let zPtr1 = kb.reshape(2, tile1Ptr)
  let zPtrTile = kb.broadcast(zPtr1, tileSizePtr)
  let zPtrs = kb.offset(zPtrTile, iota, tileSizePtr)
  kb.storePtrTko(zPtrs, zData)

  kb.ret()
  return kb.build()

# ############################################################
# 3. Structural bytecode verification
# ############################################################

proc verifyVecAddBytecode(m: BytecodeModule) =
  let bc = toBytecode(m)
  doAssert bc.len > 0
  doAssert bc[0] == 0x7F'u8
  echo "  ✓ Bytecode: ", bc.len, " bytes"

  doAssert m.functions.len == 1
  doAssert m.functions[0].name == "vec_add_kernel"
  let body = m.functions[0].body

  var checks: array[7, bool]
  for op in body:
    case op.opcode
    of OpIota:        checks[0] = true
    of OpReshape:     checks[1] = true
    of OpBroadcast:   checks[2] = true
    of OpOffset:      checks[3] = true
    of OpLoadPtrTko:  checks[4] = true
    of OpAddF:        checks[5] = true; doAssert op.operandIndices.len == 2
    of OpStorePtrTko: checks[6] = true
    else: discard

  let names = ["iota","reshape","broadcast","offset","loadPtrTko","addF","storePtrTko"]
  for i, c in checks:
    if not c:
      raise newException(AssertionError, "bytecode missing required op: " & names[i])
  echo "  ✓ All 7 required operations present (", body.len, " ops total)"

# ############################################################
# 4. Host runner
# ############################################################

proc runVecAdd*() =
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  e02: Vector Addition                               ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""

  const tileSize = 128'i64
  const totalElements = 1024'i64

  # =========================================================
  # Reference computation
  # =========================================================
  echo "[ref] z[i] = x[i] + y[i]  (CPU)"
  var hx = newSeq[float32](totalElements)
  var hy = newSeq[float32](totalElements)
  for i in 0'i64 ..< totalElements:
    hx[i] = float32(i)
    hy[i] = float32(100 + i)
  let hzRef = refVecAdd(hx, hy)
  doAssert abs(hzRef[0] - 100.0'f32) < 1e-6
  doAssert abs(hzRef[totalElements-1] -
    (totalElements-1 + 100 + totalElements-1).float32) < 1e-6
  echo "  ✓ Reference: z[0]=", hzRef[0], " z[last]=", hzRef[totalElements-1]

  # =========================================================
  # TileIR bytecode builder + verification
  # =========================================================
  echo ""
  echo "[bc]  Building TileIR kernel..."
  let m = buildVecAddKernel(tileSize)
  verifyVecAddBytecode(m)

  let bc = toBytecode(m)
  let tmp = "/tmp/cutile_examples"
  if not dirExists(tmp): createDir(tmp)
  let bcPath = tmp / "e02_vec_add.bc"
  var s = newStringOfCap(bc.len)
  for b in bc: s.add(chr(b))
  writeFile(bcPath, s)
  echo "  Bytecode dumped (", bc.len, " B)"

  # =========================================================
  # GPU execution (errors propagate — no try/catch)
  # =========================================================
  echo ""
  echo "[gpu] Trying cuModuleLoadData (TileIR JIT)…"
  let (device, ctx) = initCuda(0)
  defer: closeCuda(ctx)

  let gpuArch = getSMArch(device)
  echo "  GPU: ", gpuArch
  let cudaMod = compileBytecodeCached(m, gpuArch)
  let kernel = getFunction(cudaMod, "vec_add_kernel")
  echo "  ✓ Kernel loaded"

  let bytes = (totalElements * int64(sizeof(float32))).csize_t
  var dX = allocDevice(bytes)
  var dY = allocDevice(bytes)
  var dZ = allocDevice(bytes)
  defer: freeDevice(dX); freeDevice(dY); freeDevice(dZ)
  dX.h2d(hx[0].unsafeAddr, bytes)
  dY.h2d(hy[0].unsafeAddr, bytes)

  let grid = (totalElements div tileSize).uint32
  let args = [cast[pointer](dX.devPtr),
              cast[pointer](dY.devPtr),
              cast[pointer](dZ.devPtr)]
  launchKernel(kernel, grid, 1'u32, 1'u32, args)
  synchronize()

  var hzGpu = newSeq[float32](totalElements)
  dZ.d2h(hzGpu[0].unsafeAddr, bytes)

  if refVecAddAllClose(hx, hy, hzGpu):
    echo "  ✓ GPU result matches reference!"
  else:
    echo "  ✗ GPU result MISMATCH!"

  echo ""
  echo "✓ e02 vector addition done"

when isMainModule:
  runVecAdd()
