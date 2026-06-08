# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCL runtime execution DSL.
##
## Provides a high-level wrapper around the OpenCL C API for compiling
## and executing OpenCL C compute kernels.

import std/[os, strutils]
import workspace/crucible/src/abis/cl_abi

type
  OpenCLError* = ref object of CatchableError

  OpenCLDevice* = object
    id*: Pdevice_id

  OpenCLContext* = object
    device*: OpenCLDevice
    platform: Pplatform_id
    context: Pcontext
    commands: Pcommand_queue

  OpenCLBuffer* = object
    ctx*: OpenCLContext
    size*: int
    mem: Pmem

  OpenCLKernel* = object
    ctx: OpenCLContext
    kernel: Pkernel
    program: Pprogram

proc check(res: TClResult) =
  if res != SUCCESS:
    raise OpenCLError(msg: "OpenCL error: " & $res)

proc listPlatforms(): seq[Pplatform_id] =
  var platformCount: cl_uint
  let res = getPlatformIDs(0, nil, platformCount.addr)
  if res != SUCCESS: return
  if platformCount > 0:
    result = newSeq[Pplatform_id](platformCount.int)
    check getPlatformIDs(platformCount, result[0].addr, nil)

proc listDevices(platform: Pplatform_id, typ: TDeviceType = DEVICE_TYPE_ALL): seq[OpenCLDevice] =
  var deviceCount: cl_uint
  let res = getDeviceIDs(platform, typ, 0, nil, deviceCount.addr)
  if res == DEVICE_NOT_FOUND:
    return
  check res
  if deviceCount > 0:
    var ids = newSeq[Pdevice_id](deviceCount.int)
    check getDeviceIDs(platform, typ, deviceCount, ids[0].addr, nil)
    for id in ids:
      result.add(OpenCLDevice(id: id))

proc listDevices*(): seq[OpenCLDevice] =
  for platform in listPlatforms():
    result.add(platform.listDevices())

proc queryString(device: OpenCLDevice, info: Tdevice_info): string =
  var size: cl_size_t
  check getDeviceInfo(device.id, info, 0, nil, size.addr)
  if size > 0:
    result = newString(size.int)
    check getDeviceInfo(device.id, info, size, result[0].addr, nil)

proc name*(device: OpenCLDevice): string = device.queryString(DEVICE_NAME)
proc vendor*(device: OpenCLDevice): string = device.queryString(DEVICE_VENDOR)
proc version*(device: OpenCLDevice): string = device.queryString(DEVICE_VERSION)

proc isGpu*(device: OpenCLDevice): bool =
  var deviceType: TDeviceType
  check getDeviceInfo(device.id, DEVICE_TYPE, cl_size_t(sizeof(deviceType)), deviceType.addr, nil)
  result = (deviceType.int64 and DEVICE_TYPE_GPU.int64) != 0

proc initOpenCL*(device: OpenCLDevice): OpenCLContext =
  var
    id = device.id
    status: TClResult
  result.device = device
  result.context = createContext(nil, 1, id.addr, nil, nil, status.addr)
  check status
  result.commands = createCommandQueue(result.context, id, 0, status.addr)
  check status

proc initOpenCL*(): OpenCLContext =
  let devices = listDevices()
  if devices.len == 0:
    raise OpenCLError(msg: "No OpenCL devices found")
  # Pick first GPU; fall back to first device
  var dev = devices[0]
  for d in devices:
    if d.isGpu():
      dev = d
      break
  result = initOpenCL(dev)

proc shutdown*(ctx: var OpenCLContext) =
  discard releaseCommandQueue(ctx.commands)
  discard releaseContext(ctx.context)

proc allocBuffer*(ctx: OpenCLContext, size: int): OpenCLBuffer =
  var status: TClResult
  result.ctx = ctx
  result.size = size
  result.mem = createBuffer(ctx.context, MEM_READ_WRITE, cl_size_t(size), nil, status.addr)
  check status

proc dealloc*(buffer: var OpenCLBuffer) =
  check releaseMemObject(buffer.mem)

proc writeBuffer*(buffer: OpenCLBuffer, data: pointer, size: int) =
  if size != buffer.size:
    raise OpenCLError(msg: "Attempted to write " & $size & " bytes, but buffer size is " & $buffer.size)
  check buffer.ctx.commands.enqueueWriteBuffer(
    buffer.mem, CL_TRUE, 0, cl_size_t(size), data, 0, nil, nil
  )

proc writeBuffer*[T](buffer: OpenCLBuffer, data: openArray[T]) =
  if data.len > 0:
    buffer.writeBuffer(data[0].unsafeAddr, sizeof(T) * data.len)

proc readBuffer*[T](buffer: OpenCLBuffer, data: ptr UncheckedArray[T]) =
  check buffer.ctx.commands.enqueueReadBuffer(
    buffer.mem, CL_TRUE, 0, cl_size_t(buffer.size), data[0].addr, 0, nil, nil
  )

proc readBuffer*[T](buffer: OpenCLBuffer): seq[T] =
  if buffer.size mod sizeof(T) != 0:
    raise OpenCLError(msg: "Buffer size is not divisible by item type size")
  if buffer.size > 0:
    result = newSeq[T](buffer.size div sizeof(T))
    check buffer.ctx.commands.enqueueReadBuffer(
      buffer.mem, CL_TRUE, 0, cl_size_t(buffer.size), result[0].addr, 0, nil, nil
    )

proc compileKernel*(ctx: OpenCLContext, name: string, source: string): OpenCLKernel =
  result.ctx = ctx
  let strings = allocCStringArray([source])
  defer: deallocCStringArray(strings)

  var
    length = cl_size_t(source.len)
    status: TClResult
  let program = ctx.context.createProgramWithSource(1, strings, length.addr, status.addr)
  check status
  result.program = program

  status = buildProgram(program, 1, ctx.device.id.addr, nil, nil, nil)

  if status == BUILD_PROGRAM_FAILURE:
    var logLength: cl_size_t
    check getProgramBuildInfo(program, ctx.device.id, PROGRAM_BUILD_LOG, 0, nil, logLength.addr)
    if logLength > 0:
      var log = newString(logLength.int)
      check getProgramBuildInfo(program, ctx.device.id, PROGRAM_BUILD_LOG,
                                logLength, log[0].addr, nil)
      raise OpenCLError(msg: "Failed to build program '" & name & "': " & log)
    else:
      raise OpenCLError(msg: "Failed to build program '" & name & "'")
  else:
    check status

  result.kernel = createKernel(program, name.cstring, status.addr)
  check status

proc destroyKernel*(kernel: var OpenCLKernel) =
  discard releaseKernel(kernel.kernel)
  discard releaseProgram(kernel.program)

proc setArg*(kernel: var OpenCLKernel, index: int, buffer: OpenCLBuffer) =
  check setKernelArg(kernel.kernel, cl_uint(index), cl_size_t(sizeof(Pmem)), buffer.mem.unsafeAddr)

proc setArg*[T](kernel: var OpenCLKernel, index: int, value: T) =
  var data = value
  check setKernelArg(kernel.kernel, cl_uint(index), cl_size_t(sizeof(T)), data.addr)

proc runKernel*(kernel: OpenCLKernel, globalWorkSize, localWorkSize: openArray[cl_size_t]) =
  if globalWorkSize.len == 0:
    raise OpenCLError(msg: "Global work size must have at least one dimension")
  if globalWorkSize.len != localWorkSize.len:
    raise OpenCLError(msg: "Dimension of global work size must equal dimension of local work size")

  check enqueueNDRangeKernel(
    kernel.ctx.commands,
    kernel.kernel,
    cl_uint(globalWorkSize.len),
    nil,
    unsafeAddr globalWorkSize[0],
    unsafeAddr localWorkSize[0],
    0, nil, nil
  )
  check finish(kernel.ctx.commands)

# ═══════════════════════════════════════════════════════════════════════
# High-level helper: execOpenCL
# ═══════════════════════════════════════════════════════════════════════

proc execOpenCL*(
  ctx: OpenCLContext,
  source: string,
  entryPoint: string,
  outputBytes: int,
  inputs: openArray[tuple[data: pointer, size: int]]
): seq[byte] =
  ## Compiles and executes an OpenCL C compute kernel, returning the
  ## output buffer contents as `seq[byte]`.
  ##
  ## - `source`:     OpenCL C source code
  ## - `entryPoint`: name of the kernel entry point
  ## - `outputBytes`: number of bytes to read back as result
  ## - `inputs`:     sequence of (pointer, size) tuples for input buffers
  ##
  ## Bindings follow OpenCL kernel parameter order:
  ##   arg 0..N-1 = inputs (in order), arg N = output.
  let numInputs = inputs.len

  # Allocate input buffers + output buffer
  var inputBuffers = newSeq[OpenCLBuffer](numInputs)
  for i in 0 ..< numInputs:
    inputBuffers[i] = ctx.allocBuffer(inputs[i].size)
  defer:
    for buf in inputBuffers.mitems:
      buf.dealloc()
  var outBuf = ctx.allocBuffer(outputBytes)
  defer:
    outBuf.dealloc()
  var kernel = ctx.compileKernel(entryPoint, source)
  defer:
    kernel.destroyKernel()

  # Write input data
  for i in 0 ..< numInputs:
    inputBuffers[i].writeBuffer(inputs[i].data, inputs[i].size)

  # Set kernel args: inputs first, then output
  for i in 0 ..< numInputs:
    kernel.setArg(i, inputBuffers[i])
  kernel.setArg(numInputs, outBuf)

  let gs = [cl_size_t(1)]
  let ls = [cl_size_t(1)]
  kernel.runKernel(gs, ls)

  # Read output
  result = newSeq[byte](outputBytes)
  check outBuf.ctx.commands.enqueueReadBuffer(
    outBuf.mem, CL_TRUE, 0, cl_size_t(outputBytes), result[0].addr, 0, nil, nil
  )
# ═══════════════════════════════════════════════════════════════════════
# Even higher-level combined: create context + exec + shutdown
# ═══════════════════════════════════════════════════════════════════════

proc execOpenCL*(
  source: string,
  entryPoint: string,
  outputBytes: int,
  inputs: openArray[tuple[data: pointer, size: int]]
): seq[byte] =
  ## Convenience overload that creates a temporary OpenCL context.
  var ctx = initOpenCL()
  defer: ctx.shutdown()
  result = ctx.execOpenCL(source, entryPoint, outputBytes, inputs)
