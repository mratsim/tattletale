## cutile/examples/e03_saxpy.nim
## Port of cutile-rs tutorial 03-saxpy
##
## SAXPY: Single-precision A·X Plus Y (in-place).
##   y = alpha * x + y
##
## Demonstrates:
##   - Broadcasting a scalar to tile shape
##   - In-place read-modify-write (load y, compute, store y)
##   - FMA (fused multiply-add) for efficiency

import
  std/[os, strutils],
  ../bytecode,
  ../dsl,
  ../compiler

# ############################################################
# Kernel builder
# ############################################################

proc buildSaxpyKernel*(tileSize: int64): BytecodeModule =
  ## Build an in-place SAXPY kernel: y = alpha * x + y
  ##
  ## Kernel inputs (TileIR ABI):
  ##   [0] ptr to x  (pointer)
  ##   [1] ptr to y  (pointer, in-place: read AND write)
  ##   [2] alpha     (float32 scalar)
  ##
  ## Body (per tile block):
  ##   offsets = iota(tileSize)
  ##   xPtr    = broadcast(reshape(x_base, [1]), [tileSize]) + offsets
  ##   xTile   = loadPtrTko(xPtr)
  ##   yPtr    = broadcast(reshape(y_base, [1]), [tileSize]) + offsets
  ##   yTile   = loadPtrTko(yPtr)
  ##   alphaTile = broadcast(alpha, [tileSize])
  ##   result = fma(alphaTile, xTile, yTile)   # alpha * x + y
  ##   storePtrTko(yPtr, result)
  let ptrF32 = TileType(shape: @[], elemType: ElemPointer)
  let scalarF32 = TileType(shape: @[], elemType: ElemF32)
  let tileF32 = TileType(shape: @[tileSize], elemType: ElemF32)
  let tilePtr = TileType(shape: @[tileSize], elemType: ElemPointer)
  let tile1Ptr = TileType(shape: @[1], elemType: ElemPointer)
  let tileI32 = TileType(shape: @[tileSize], elemType: ElemI32)

  let kb = newKernel(
    "saxpy_kernel",
    @[ptrF32, ptrF32, scalarF32],
    @[]
  )

  # Offsets
  let offsets = kb.iota(@[tileSize], ElemI32)

  # ---- Load x ----
  let xPtr1 = kb.reshape(0, tile1Ptr)
  let xPtrTile = kb.broadcast(xPtr1, tilePtr)
  let xPtrs = kb.offset(xPtrTile, offsets, tilePtr)
  let (xData, _) = kb.loadPtrTko(xPtrs, tileF32)

  # ---- Load y (in-place) ----
  let yPtr1 = kb.reshape(1, tile1Ptr)
  let yPtrTile = kb.broadcast(yPtr1, tilePtr)
  let yPtrs = kb.offset(yPtrTile, offsets, tilePtr)
  let (yData, _) = kb.loadPtrTko(yPtrs, tileF32)

  # ---- Broadcast alpha ----
  let alphaTile = kb.broadcast(2, tileF32)

  # ---- Compute: y = alpha * x + y ----
  let result = kb.fma(alphaTile, xData, yData, tileF32)

  # ---- Store back to y (in-place) ----
  kb.storePtrTko(yPtrs, result)

  kb.ret()
  return kb.build()

# ############################################################
# Verification
# ############################################################

proc verifySaxpyBytecode*(m: BytecodeModule) =
  let bc = toBytecode(m)
  doAssert bc.len > 0
  doAssert bc[0] == 0x7F'u8
  echo "  ✓ Bytecode: ", bc.len, " bytes"

  doAssert m.functions.len == 1
  doAssert m.functions[0].name == "saxpy_kernel"

  let body = m.functions[0].body
  var foundFma = false
  var foundBroadcast = false
  for op in body:
    if op.opcode == OpFma:
      foundFma = true
      doAssert op.operandIndices.len == 3
    if op.opcode == OpBroadcast:
      foundBroadcast = true
  doAssert foundFma, "Missing OpFma in SAXPY kernel"
  doAssert foundBroadcast, "Missing OpBroadcast in SAXPY kernel"
  echo "  ✓ Kernel has ", body.len, " ops, includes FMA + broadcast"

# ############################################################
# Host runner
# ############################################################

proc runSaxpy*() =
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  e03: SAXPY (alpha * x + y)                         ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""

  const tileSize = 128'i64
  let m = buildSaxpyKernel(tileSize)
  verifySaxpyBytecode(m)

  let bc = toBytecode(m)
  let tmp = "/tmp/cutile_examples"
  if not dirExists(tmp): createDir(tmp)
  let path = tmp / "e03_saxpy.bc"
  var s = newStringOfCap(bc.len)
  for b in bc: s.add(chr(b))
  writeFile(path, s)
  echo "  [dbg] bc.len=", bc.len, " s.len=", s.len
  echo "  Wrote bytecode to: ", path
  echo ""

  # ── GPU compilation & launch ──
  echo "[gpu] Compiling + launching..."
  let sm = "sm_120"
  discard compileBytecodeCached(m, sm)

when isMainModule:
  runSaxpy()
