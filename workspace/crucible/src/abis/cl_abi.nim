# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
import std/macros

# ═══════════════════════════════════════════════════════════════════════
# Scalar types
# ═══════════════════════════════════════════════════════════════════════

type
  cl_int*      = int32
  cl_uint*     = uint32
  cl_ulong*    = uint64
  cl_size_t*   = csize_t
  cl_bool*     = cl_uint

# ═══════════════════════════════════════════════════════════════════════
# Opaque pointer types
# ═══════════════════════════════════════════════════════════════════════

type
  Pplatform_id*   = pointer
  Pdevice_id*     = pointer
  Pcontext*       = pointer
  Pcommand_queue* = pointer
  Pprogram*       = pointer
  Pkernel*        = pointer
  Pmem*           = pointer

  TDeviceType*    = cl_ulong
  Tdevice_info*   = cl_uint
  TClResult*      = cl_int

# ═══════════════════════════════════════════════════════════════════════
# Error codes
# ═══════════════════════════════════════════════════════════════════════

const
  SUCCESS*              = TClResult(0)
  DEVICE_NOT_FOUND*     = TClResult(-1)
  BUILD_PROGRAM_FAILURE* = TClResult(-11)
  INVALID_ARG_INDEX*     = TClResult(-49)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

const
  CL_TRUE*  = cl_bool(1)
  CL_FALSE* = cl_bool(0)

  # Device types
  DEVICE_TYPE_ALL*   = TDeviceType(0xFFFFFFFF)
  DEVICE_TYPE_GPU*   = TDeviceType(1 shl 2)

  # Device info (values per /usr/include/CL/cl.h)
  DEVICE_NAME*             = Tdevice_info(0x102B)
  DEVICE_VENDOR*           = Tdevice_info(0x102C)
  DEVICE_VERSION*          = Tdevice_info(0x102F)
  DEVICE_TYPE*             = Tdevice_info(0x1000)

  # Memory flags
  MEM_READ_WRITE*  = cl_ulong(1 shl 0)
  MEM_WRITE_ONLY*  = cl_ulong(1 shl 1)
  MEM_READ_ONLY*   = cl_ulong(1 shl 2)

  # Program build info
  PROGRAM_BUILD_LOG* = 0x1183

  # Command queue properties
  QUEUE_PROFILING_ENABLE* = cl_ulong(1 shl 1)

# ═══════════════════════════════════════════════════════════════════════
# Library loading
# ═══════════════════════════════════════════════════════════════════════

const ClLib* = (
  when defined(linux):   "libOpenCL.so"
  elif defined(macosx):  "/System/Library/Frameworks/OpenCL.framework/OpenCL"
  elif defined(windows): "OpenCL.dll"
  else:                  "libOpenCL.so"
)

# ═══════════════════════════════════════════════════════════════════════
# Platform API
# ═══════════════════════════════════════════════════════════════════════

proc getPlatformIDs*(
  num_entries: cl_uint,
  platforms: Pplatform_id,
  num_platforms: ptr cl_uint
): TClResult {.importc: "clGetPlatformIDs", dynlib: ClLib.}

# ═══════════════════════════════════════════════════════════════════════
# Device API
# ═══════════════════════════════════════════════════════════════════════

proc getDeviceIDs*(
  platform: Pplatform_id,
  device_type: TDeviceType,
  num_entries: cl_uint,
  devices: Pdevice_id,
  num_devices: ptr cl_uint
): TClResult {.importc: "clGetDeviceIDs", dynlib: ClLib.}

proc getDeviceInfo*(
  device: Pdevice_id,
  param_name: Tdevice_info,
  param_value_size: cl_size_t,
  param_value: pointer,
  param_value_size_ret: ptr cl_size_t
): TClResult {.importc: "clGetDeviceInfo", dynlib: ClLib.}

# ═══════════════════════════════════════════════════════════════════════
# Context API
# ═══════════════════════════════════════════════════════════════════════

type
  PContextCallback* = proc (name: cstring, errorInfo: cstring, privateInfo: pointer, cb: cl_size_t, userData: pointer) {.cdecl.}

proc createContext*(
  properties: pointer,
  num_devices: cl_uint,
  devices: ptr Pdevice_id,
  pfn_notify: PContextCallback,
  user_data: pointer,
  errcode_ret: ptr TClResult
): Pcontext {.importc: "clCreateContext", dynlib: ClLib.}

# ═══════════════════════════════════════════════════════════════════════
# Command Queue API
# ═══════════════════════════════════════════════════════════════════════

proc createCommandQueue*(
  context: Pcontext,
  device: Pdevice_id,
  properties: cl_ulong,
  errcode_ret: ptr TClResult
): Pcommand_queue {.importc: "clCreateCommandQueue", dynlib: ClLib.}

# ═══════════════════════════════════════════════════════════════════════
# Memory Object API
# ═══════════════════════════════════════════════════════════════════════

proc createBuffer*(
  context: Pcontext,
  flags: cl_ulong,
  size: cl_size_t,
  host_ptr: pointer,
  errcode_ret: ptr TClResult
): Pmem {.importc: "clCreateBuffer", dynlib: ClLib.}

proc releaseMemObject*(
  memobj: Pmem
): TClResult {.importc: "clReleaseMemObject", dynlib: ClLib.}

proc enqueueWriteBuffer*(
  command_queue: Pcommand_queue,
  buffer: Pmem,
  blocking_write: cl_bool,
  offset: cl_size_t,
  size: cl_size_t,
  ptr_data: pointer,
  num_events_in_wait_list: cl_uint,
  event_wait_list: pointer,
  event: pointer
): TClResult {.importc: "clEnqueueWriteBuffer", dynlib: ClLib.}

proc enqueueReadBuffer*(
  command_queue: Pcommand_queue,
  buffer: Pmem,
  blocking_read: cl_bool,
  offset: cl_size_t,
  size: cl_size_t,
  ptr_data: pointer,
  num_events_in_wait_list: cl_uint,
  event_wait_list: pointer,
  event: pointer
): TClResult {.importc: "clEnqueueReadBuffer", dynlib: ClLib.}

proc enqueueFillBuffer*(
  command_queue: Pcommand_queue,
  buffer: Pmem,
  pattern: pointer,
  pattern_size: cl_size_t,
  offset: cl_size_t,
  size: cl_size_t,
  num_events_in_wait_list: cl_uint,
  event_wait_list: pointer,
  event: pointer
): TClResult {.importc: "clEnqueueFillBuffer", dynlib: ClLib.}

# ═══════════════════════════════════════════════════════════════════════
# Program & Kernel API
# ═══════════════════════════════════════════════════════════════════════

proc createProgramWithSource*(
  context: Pcontext,
  count: cl_uint,
  strings: cstringArray,
  lengths: ptr cl_size_t,
  errcode_ret: ptr TClResult
): Pprogram {.importc: "clCreateProgramWithSource", dynlib: ClLib.}

proc buildProgram*(
  program: Pprogram,
  num_devices: cl_uint,
  device_list: Pdevice_id,
  options: cstring,
  pfn_notify: pointer,
  user_data: pointer
): TClResult {.importc: "clBuildProgram", dynlib: ClLib.}

proc getProgramBuildInfo*(
  program: Pprogram,
  device: Pdevice_id,
  param_name: cl_uint,
  param_value_size: cl_size_t,
  param_value: pointer,
  param_value_size_ret: ptr cl_size_t
): TClResult {.importc: "clGetProgramBuildInfo", dynlib: ClLib.}

proc releaseProgram*(
  program: Pprogram
): TClResult {.importc: "clReleaseProgram", dynlib: ClLib.}

proc createKernel*(
  program: Pprogram,
  kernel_name: cstring,
  errcode_ret: ptr TClResult
): Pkernel {.importc: "clCreateKernel", dynlib: ClLib.}

proc releaseKernel*(
  kernel: Pkernel
): TClResult {.importc: "clReleaseKernel", dynlib: ClLib.}

proc setKernelArg*(
  kernel: Pkernel,
  arg_index: cl_uint,
  arg_size: cl_size_t,
  arg_value: pointer
): TClResult {.importc: "clSetKernelArg", dynlib: ClLib.}

# ═══════════════════════════════════════════════════════════════════════
# Execution API
# ═══════════════════════════════════════════════════════════════════════

proc enqueueNDRangeKernel*(
  command_queue: Pcommand_queue,
  kernel: Pkernel,
  work_dim: cl_uint,
  global_work_offset: pointer,
  global_work_size: ptr cl_size_t,
  local_work_size: ptr cl_size_t,
  num_events_in_wait_list: cl_uint,
  event_wait_list: pointer,
  event: pointer
): TClResult {.importc: "clEnqueueNDRangeKernel", dynlib: ClLib.}

# ═══════════════════════════════════════════════════════════════════════
# Flush / Finish
# ═══════════════════════════════════════════════════════════════════════

proc finish*(
  command_queue: Pcommand_queue
): TClResult {.importc: "clFinish", dynlib: ClLib.}

proc flush*(
  command_queue: Pcommand_queue
): TClResult {.importc: "clFlush", dynlib: ClLib.}

# ═══════════════════════════════════════════════════════════════════════
# Context & Queue Release
# ═══════════════════════════════════════════════════════════════════════

proc releaseContext*(
  context: Pcontext
): TClResult {.importc: "clReleaseContext", dynlib: ClLib.}

proc releaseCommandQueue*(
  command_queue: Pcommand_queue
): TClResult {.importc: "clReleaseCommandQueue", dynlib: ClLib.}
