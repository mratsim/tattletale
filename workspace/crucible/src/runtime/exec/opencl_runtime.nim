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
##
## Example (engine API):
##   import workspace/crucible
##   const code = opencl:
##     proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
##       output[0] = a[0] + b[0]
##   var engine = bkOpenCL.init()
##   engine.ingest(code)
##   var out: array[1, uint32]
##   engine.run("addKernel", out, ([1'u32], [2'u32]))


import workspace/crucible/src/abis/cl_abi

type

  OpenCLDevice* = object
    id: Pdevice_id

  OpenCLContext* = object
    device*: OpenCLDevice
    platform: Pplatform_id
    context: Pcontext
    commands: Pcommand_queue

  OpenCLBuffer* = object
    ctx: OpenCLContext
    size: int
    mem: Pmem

  OpenCLKernel* = object
    ctx: OpenCLContext
    kernel: Pkernel
    program: Pprogram

template check*(res: TClResult) =
  ## Unified error policy: stacktrace + stderr + quit(1).
  ## No exceptions as the public contract.
  ## A template so instantiationInfo() reports the caller's location.
  let code = res
  if code != SUCCESS:
    writeStackTrace()
    stderr.write($instantiationInfo() & " exited with error: OpenCL error " & $code & '\n')
    quit 1

proc listPlatforms(): seq[Pplatform_id] =
  var platformCount: cl_uint
  let res = getPlatformIDs(0, nil, platformCount.addr)
  if res != SUCCESS: return
  if platformCount > 0:
    result = newSeq[Pplatform_id](platformCount.int)
    check getPlatformIDs(platformCount, result[0].addr, nil)

proc listDevices(platform: Pplatform_id): seq[OpenCLDevice] =
  var deviceCount: cl_uint
  let res = getDeviceIDs(platform, DEVICE_TYPE_ALL, 0, nil, deviceCount.addr)
  if res == DEVICE_NOT_FOUND:
    return
  check res
  if deviceCount > 0:
    var ids = newSeq[Pdevice_id](deviceCount.int)
    check getDeviceIDs(platform, DEVICE_TYPE_ALL, deviceCount, ids[0].addr, nil)
    for id in ids:
      result.add(OpenCLDevice(id: id))

proc listDevices(): seq[OpenCLDevice] =
  for platform in listPlatforms():
    result.add(platform.listDevices())

proc queryString(device: OpenCLDevice, info: Tdevice_info): string =
  var size: cl_size_t
  check getDeviceInfo(device.id, info, 0, nil, size.addr)
  if size > 0:
    result = newString(size.int)
    check getDeviceInfo(device.id, info, size, result[0].addr, nil)
    if result[^1] == '\0':   # clGetDeviceInfo strings are NUL-terminated
      result.setLen(result.len - 1)

proc name*(device: OpenCLDevice): string = device.queryString(DEVICE_NAME)

proc isGpu(device: OpenCLDevice): bool =
  var deviceType: TDeviceType
  check getDeviceInfo(device.id, DEVICE_TYPE, cl_size_t(sizeof(deviceType)), deviceType.addr, nil)
  result = (deviceType.int64 and DEVICE_TYPE_GPU.int64) != 0

proc initOpenCL(device: OpenCLDevice): OpenCLContext =
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
    quit("OpenCL: no devices found")
  # Pick first GPU; fall back to first device
  var dev = devices[0]
  for d in devices:
    if d.isGpu():
      dev = d
      break
  result = initOpenCL(dev)

proc shutdown*(ctx: var OpenCLContext) =
  ## Idempotent: safe to call multiple times (manual shutdown + =destroy).
  if ctx.context != nil:
    discard releaseCommandQueue(ctx.commands)
    discard releaseContext(ctx.context)
    ctx.commands = nil
    ctx.context = nil

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
    quit("OpenCL writeBuffer: attempted to write " & $size & " bytes, but buffer size is " & $buffer.size)
  check buffer.ctx.commands.enqueueWriteBuffer(
    buffer.mem, CL_TRUE, 0, cl_size_t(size), data, 0, nil, nil
  )



proc readBuffer*(buffer: OpenCLBuffer, data: pointer, size: int) =
  ## Raw readback into a caller-provided buffer (used by the engines).
  if size > buffer.size:
    quit("OpenCL readBuffer: read of " & $size & " bytes exceeds buffer size " & $buffer.size)
  check buffer.ctx.commands.enqueueReadBuffer(
    buffer.mem, CL_TRUE, 0, cl_size_t(size), data, 0, nil, nil
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
      quit("OpenCL failed to build program '" & name & "': " & log)
    else:
      quit("OpenCL failed to build program '" & name & "'")
  else:
    check status

  result.kernel = createKernel(program, name.cstring, status.addr)
  check status

proc destroyKernel*(kernel: var OpenCLKernel) =
  discard releaseKernel(kernel.kernel)
  discard releaseProgram(kernel.program)

proc setArg*(kernel: var OpenCLKernel, index: int, buffer: OpenCLBuffer) =
  check setKernelArg(kernel.kernel, cl_uint(index), cl_size_t(sizeof(Pmem)), buffer.mem.unsafeAddr)


proc runKernel*(kernel: OpenCLKernel, globalWorkSize, localWorkSize: openArray[cl_size_t]) =
  if globalWorkSize.len == 0:
    quit("OpenCL runKernel: global work size must have at least one dimension")
  if globalWorkSize.len != localWorkSize.len:
    quit("OpenCL runKernel: dimension of global work size must equal dimension of local work size")

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

proc setArg*(kernel: var OpenCLKernel, index: int, size: int, data: pointer) =
  ## Bind a scalar kernel argument by value (e.g. alpha/beta coefficients).
  check setKernelArg(kernel.kernel, cl_uint(index), cl_size_t(size), data)
