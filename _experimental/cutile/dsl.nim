# cutile/dsl.nim
# Layer 4: Nim DSL for TileIR kernels

import
  std/[tables],
  bytecode

# ############################################################
# Kernel builder
# ############################################################

type
  KernelBuilder* = ref object
    module*: BytecodeModule
    funcName*: string
    funcType*: FuncType
    body*: seq[BytecodeOp]
    numResults*: int

proc newKernel*(name: string,
                inputs: seq[TileType],
                results: seq[TileType] = @[]): KernelBuilder =
  result = KernelBuilder(
    module: newBytecodeModule(),
    funcName: name,
    funcType: FuncType(inputs: inputs, results: results),
    body: @[],
    numResults: inputs.len
  )

proc emit*(kb: KernelBuilder, op: BytecodeOp) =
  kb.body.add(op)
  kb.numResults += op.resultTypes.len

proc build*(kb: KernelBuilder): BytecodeModule =
  let bcFunc = BytecodeFunction(
    name: kb.funcName,
    funcType: kb.funcType,
    body: kb.body
  )
  kb.module.functions.add(bcFunc)
  return kb.module

# ############################################################
# Tile operations
# ############################################################

proc iota*(kb: KernelBuilder, shape: seq[int64],
           elemType: TileElemType = ElemI32): ValueIdx =
  let resIdx = kb.numResults
  let tileT = TileType(shape: shape, elemType: elemType)
  kb.emit(BytecodeOp(
    opcode: OpIota,
    resultTypes: @[tileT],
    operandIndices: @[],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc getTileBlockId*(kb: KernelBuilder): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpGetTileBlockId,
    resultTypes: @[TileType(shape: @[], elemType: ElemI32)],
    operandIndices: @[],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc addF*(kb: KernelBuilder, a, b: ValueIdx,
           resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpAddF,
    resultTypes: @[resultType],
    operandIndices: @[a, b],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc mulF*(kb: KernelBuilder, a, b: ValueIdx,
           resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpMulF,
    resultTypes: @[resultType],
    operandIndices: @[a, b],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc fma*(kb: KernelBuilder, a, b, c: ValueIdx,
          resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpFma,
    resultTypes: @[resultType],
    operandIndices: @[a, b, c],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc broadcast*(kb: KernelBuilder, operand: ValueIdx,
                resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpBroadcast,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc reshape*(kb: KernelBuilder, operand: ValueIdx,
              resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpReshape,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc offset*(kb: KernelBuilder, ptrOperand, indexOperand: ValueIdx,
             resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpOffset,
    resultTypes: @[resultType],
    operandIndices: @[ptrOperand, indexOperand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc constant*(kb: KernelBuilder, value: float,
               resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpConstant,
    resultTypes: @[resultType],
    operandIndices: @[],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc constant*(kb: KernelBuilder, value: int32,
               resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpConstant,
    resultTypes: @[resultType],
    operandIndices: @[],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc loadPtrTko*(kb: KernelBuilder, ptrTile: ValueIdx,
                 resultType: TileType): tuple[data: ValueIdx, token: ValueIdx] =
  let dataIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpLoadPtrTko,
    resultTypes: @[resultType, TileType(shape: @[], elemType: ElemI1)],
    operandIndices: @[ptrTile],
    attrs: initTable[string, seq[byte]]()
  ))
  let tokenIdx = kb.numResults
  return (dataIdx, tokenIdx)

proc storePtrTko*(kb: KernelBuilder, ptrTile, value: ValueIdx) =
  kb.emit(BytecodeOp(
    opcode: OpStorePtrTko,
    resultTypes: @[],
    operandIndices: @[ptrTile, value],
    attrs: initTable[string, seq[byte]]()
  ))

proc ret*(kb: KernelBuilder) =
  kb.emit(BytecodeOp(
    opcode: OpReturn,
    resultTypes: @[],
    operandIndices: @[],
    attrs: initTable[string, seq[byte]]()
  ))

proc subF*(kb: KernelBuilder, a, b: ValueIdx,
           resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpSubF,
    resultTypes: @[resultType],
    operandIndices: @[a, b],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc divF*(kb: KernelBuilder, a, b: ValueIdx,
           resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpDivF,
    resultTypes: @[resultType],
    operandIndices: @[a, b],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

# ############################################################
# Unary arithmetic ops
# ############################################################

proc negF*(kb: KernelBuilder, operand: ValueIdx,
               resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpNegF,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc absF*(kb: KernelBuilder, operand: ValueIdx,
            resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpAbsF,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc expOp*(kb: KernelBuilder, operand: ValueIdx,
             resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpExp,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc exp2Op*(kb: KernelBuilder, operand: ValueIdx,
              resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpExp2,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc logOp*(kb: KernelBuilder, operand: ValueIdx,
             resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpLog,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc log2Op*(kb: KernelBuilder, operand: ValueIdx,
              resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpLog2,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc sqrtOp*(kb: KernelBuilder, operand: ValueIdx,
              resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpSqrt,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc sinOp*(kb: KernelBuilder, operand: ValueIdx,
             resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpSin,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc cosOp*(kb: KernelBuilder, operand: ValueIdx,
             resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpCos,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc floorOp*(kb: KernelBuilder, operand: ValueIdx,
               resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpFloor,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc ceilOp*(kb: KernelBuilder, operand: ValueIdx,
              resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpCeil,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

# ############################################################
# Matrix multiply-accumulate
# ############################################################

proc mmaF*(kb: KernelBuilder, a, b, c: ValueIdx,
             resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpMmaF,
    resultTypes: @[resultType],
    operandIndices: @[a, b, c],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

proc mmaI*(kb: KernelBuilder, a, b, c: ValueIdx,
             resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  kb.emit(BytecodeOp(
    opcode: OpMmaI,
    resultTypes: @[resultType],
    operandIndices: @[a, b, c],
    attrs: initTable[string, seq[byte]]()
  ))
  return resIdx

# ############################################################
# Reduce operations (TileIR MakeRangeReduce)
# ############################################################
## NOTE: These are placeholder wrappers. TileIR reductions use
## OpMakeRangeReduce which creates a reduction range. The actual
## reduction semantics (max vs sum) and axis are encoded in attrs.

proc reduceMax*(kb: KernelBuilder, operand: ValueIdx,
                 axis: int32, resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  var attrs = initTable[string, seq[byte]]()
  # encode reduction kind and axis
  attrs["kind"] = @[0x00'u8]
  attrs["axis"] = @[axis.uint8]
  kb.emit(BytecodeOp(
    opcode: OpMakeRangeReduce,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: attrs
  ))
  return resIdx

proc reduceSum*(kb: KernelBuilder, operand: ValueIdx,
                 axis: int32, resultType: TileType): ValueIdx =
  let resIdx = kb.numResults
  var attrs = initTable[string, seq[byte]]()
  attrs["kind"] = @[0x01'u8]
  attrs["axis"] = @[axis.uint8]
  kb.emit(BytecodeOp(
    opcode: OpMakeRangeReduce,
    resultTypes: @[resultType],
    operandIndices: @[operand],
    attrs: attrs
  ))
  return resIdx

# ############################################################
# Debug print
# ############################################################

proc printTko*(kb: KernelBuilder, operand: ValueIdx, format: string = "") =
  ## Emit a print operation for debugging.
  var attrs = initTable[string, seq[byte]]()
  if format.len > 0:
    attrs["format"] = @[]
    for c in format:
      attrs["format"].add(c.uint8)
  kb.emit(BytecodeOp(
    opcode: OpPrintTko,
    resultTypes: @[],
    operandIndices: @[operand],
    attrs: attrs
  ))

# ############################################################
# Helper: build AXPY kernel
# ############################################################

proc buildAxpYKernel*(tileSize: int64): BytecodeModule =
  let ptrF32 = TileType(shape: @[], elemType: ElemPointer)
  let scalarF32 = TileType(shape: @[], elemType: ElemF32)
  let tileF32 = TileType(shape: @[tileSize], elemType: ElemF32)
  let tilePtr = TileType(shape: @[tileSize], elemType: ElemPointer)
  let tile1Ptr = TileType(shape: @[1], elemType: ElemPointer)
  
  let kb = newKernel(
    "axpy_kernel",
    @[ptrF32, ptrF32, scalarF32],
    @[]
  )
  
  # bid = getTileBlockId()
  let bid = kb.getTileBlockId()
  
  # iota = [0, 1, 2, ..., tileSize-1]
  let iota = kb.iota(@[tileSize], ElemI32)
  
  # x_ptr: reshape to 1xptr, broadcast to tilexptr
  let xPtr1 = kb.reshape(0, tile1Ptr)
  let xPtrTile = kb.broadcast(xPtr1, tilePtr)
  let xPtrs = kb.offset(xPtrTile, iota, tilePtr)
  
  # load x
  let (xData, _) = kb.loadPtrTko(xPtrs, tileF32)
  
  # y_ptr: same pattern
  let yPtr1 = kb.reshape(1, tile1Ptr)
  let yPtrTile = kb.broadcast(yPtr1, tilePtr)
  let yPtrs = kb.offset(yPtrTile, iota, tilePtr)
  
  # load y
  let (yData, _) = kb.loadPtrTko(yPtrs, tileF32)
  
  # result = alpha * x + y
  let alphaBroadcast = kb.broadcast(2, tileF32)
  let result = kb.fma(alphaBroadcast, xData, yData, tileF32)
  
  # store back to y
  kb.storePtrTko(yPtrs, result)
  
  kb.ret()
  
  return kb.build()

# ############################################################
# Helper: build simple print kernel
# ############################################################

proc buildPrintKernel*(tileSize: int64 = 128): BytecodeModule =
  let ptrF32 = TileType(shape: @[], elemType: ElemPointer)
  let tileF32 = TileType(shape: @[tileSize], elemType: ElemF32)
  let tileI32 = TileType(shape: @[tileSize], elemType: ElemI32)
  let tilePtr = TileType(shape: @[tileSize], elemType: ElemPointer)
  let tile1Ptr = TileType(shape: @[1], elemType: ElemPointer)
  
  let kb = newKernel(
    "print_kernel",
    @[ptrF32],
    @[]
  )
  
  let offsets = kb.iota(@[tileSize], ElemI32)
  
  let dataPtr1 = kb.reshape(0, tile1Ptr)
  let dataPtrTile = kb.broadcast(dataPtr1, tilePtr)
  let dataPtrs = kb.offset(dataPtrTile, offsets, tilePtr)
  
  let (data, _) = kb.loadPtrTko(dataPtrs, tileF32)
  
  kb.ret()
  
  return kb.build()
