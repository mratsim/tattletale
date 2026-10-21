# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible/src/abis/opencl_abi
import ../errors

type

  OpenCLDevice* = object
    id: Pdevice_id

  OpenCLContext* = ref OpenCLContextObj
  OpenCLContextObj = object
    device*: OpenCLDevice
    context: Pcontext
    commands: Pcommand_queue

  OpenCLBuffer* = object
    commands: Pcommand_queue
    size: int
    mem: Pmem

  OpenCLKernel* = ref OpenCLKernelObj
  OpenCLKernelObj = object
    commands: Pcommand_queue
    kernel: Pkernel
    program: Pprogram


proc `=copy`(dst: var OpenCLContextObj, src: OpenCLContextObj) {.error:
    "OpenCLContext is ref-only, copy the ref not the object".}

proc `=copy`(dst: var OpenCLKernelObj, src: OpenCLKernelObj) {.error:
    "OpenCLKernel is ref-only, copy the ref not the object".}

proc `=destroy`(ctx: var OpenCLContextObj) =
  ## Releases the OpenCL context and command queue, nil-guarded against double teardown.
  if ctx.context != nil:
    discard releaseCommandQueue(ctx.commands)
    discard releaseContext(ctx.context)
    ctx.commands = nil
    ctx.context = nil

proc `=destroy`(k: var OpenCLKernelObj) =
  ## Releases the kernel and its program, nil fields skip.
  if k.kernel != nil:
    discard releaseKernel(k.kernel)
  if k.program != nil:
    discard releaseProgram(k.program)

template check*(res: TClResult) =
  ## Unified error policy: stacktrace + stderr + quit(1).
  ## No exceptions as the public contract.
  ## A template so instantiationInfo() reports the caller's location.
  let code = res
  if code != SUCCESS:
    failLoud("OpenCL error " & $code)

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

proc newOpenCLContext(device: OpenCLDevice): OpenCLContext =
  result = OpenCLContext()
  var
    id = device.id
    status: TClResult
  result.device = device
  result.context = createContext(nil, 1, id.addr, nil, nil, status.addr)
  check status
  result.commands = createCommandQueue(result.context, id, 0, status.addr)
  check status

proc newOpenCLContext*(): OpenCLContext =
  let devices = listDevices()
  if devices.len == 0:
    failLoud("OpenCL: no devices found")
  # Pick first GPU, fall back to first device
  var dev = devices[0]
  for d in devices:
    if d.isGpu():
      dev = d
      break
  result = newOpenCLContext(dev)

proc allocBuffer*(ctx: OpenCLContext, size: int): OpenCLBuffer =
  var status: TClResult
  result.commands = ctx.commands
  result.size = size
  result.mem = createBuffer(ctx.context, MEM_READ_WRITE, cl_size_t(size), nil, status.addr)
  check status

proc dealloc*(buffer: var OpenCLBuffer) =
  check releaseMemObject(buffer.mem)

proc writeBuffer*(buffer: OpenCLBuffer, data: pointer, size: int) =
  if size != buffer.size:
    failLoud("OpenCL writeBuffer: attempted to write " & $size & " bytes, but buffer size is " & $buffer.size)
  check buffer.commands.enqueueWriteBuffer(
    buffer.mem, CL_TRUE, 0, cl_size_t(size), data, 0, nil, nil
  )

proc readBuffer*(buffer: OpenCLBuffer, data: pointer, size: int) =
  ## Raw readback into a caller-provided buffer (used by the engines).
  if size > buffer.size:
    failLoud("OpenCL readBuffer: read of " & $size & " bytes exceeds buffer size " & $buffer.size)
  check buffer.commands.enqueueReadBuffer(
    buffer.mem, CL_TRUE, 0, cl_size_t(size), data, 0, nil, nil
  )

proc compileKernel*(ctx: OpenCLContext, name: string, source: string): OpenCLKernel =
  result = OpenCLKernel()
  result.commands = ctx.commands
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
      failLoud("OpenCL failed to build program '" & name & "': " & log)
    else:
      failLoud("OpenCL failed to build program '" & name & "'")
  else:
    check status

  result.kernel = createKernel(program, name.cstring, status.addr)
  check status

proc setArg*(kernel: var OpenCLKernel, index: int, buffer: OpenCLBuffer) =
  check setKernelArg(kernel.kernel, cl_uint(index), cl_size_t(sizeof(Pmem)), buffer.mem.unsafeAddr)

proc runKernel*(kernel: OpenCLKernel, globalWorkSize, localWorkSize: openArray[cl_size_t]) =
  if globalWorkSize.len == 0:
    failLoud("OpenCL runKernel: global work size must have at least one dimension")
  if globalWorkSize.len != localWorkSize.len:
    failLoud("OpenCL runKernel: dimension of global work size must equal dimension of local work size")

  check enqueueNDRangeKernel(
    kernel.commands,
    kernel.kernel,
    cl_uint(globalWorkSize.len),
    nil,
    unsafeAddr globalWorkSize[0],
    unsafeAddr localWorkSize[0],
    0, nil, nil
  )
  check finish(kernel.commands)

proc setArg*(kernel: var OpenCLKernel, index: int, size: int, data: pointer) =
  ## Binds a scalar kernel argument by value (such as alpha/beta coefficients)
  check setKernelArg(kernel.kernel, cl_uint(index), cl_size_t(size), data)
