## cutile/runtime.nim
## Layer 3: Kernel Runtime (compile + load + launch)
##
## Combines bytecode compilation and kernel loading.
## Supports both:
##   - Direct cuModuleLoadData (CUDA 13.1+ JIT compiles TileIR)
##   - tileiras → cubin → cuModuleLoad (fallback)

import
  std/[os, strutils, tables],
  cuda_driver,
  compiler,
  bytecode

# ############################################################
# Kernel handle
# ############################################################

type
  KernelHandle* = ref object
    module*: CUmodule
    function*: CUfunction
    funcName*: string

proc loadKernelFromModule*(cudaModule: CUmodule, funcName: string): KernelHandle =
  ## Create a KernelHandle from an already-loaded CUmodule.
  result = KernelHandle(
    module: cudaModule,
    function: getFunction(cudaModule, funcName),
    funcName: funcName
  )
proc loadKernelFromCubin*(cubinPath: string, funcName: string): KernelHandle =
  ## Load a kernel from a compiled cubin file.
  let m = loadModuleFromFile(cubinPath)
  result = loadKernelFromModule(m, funcName)
proc unload*(k: var KernelHandle) =
  if pointer(k.module) != nil:
    unloadModule(k.module)
    k.module = cast[CUmodule](nil)

proc launch*(k: KernelHandle, gridX: uint32 = 1,
             gridY: uint32 = 1, gridZ: uint32 = 1,
             args: openArray[pointer] = @[]) =
  launchKernel(k.function, gridX, gridY, gridZ, args)

# ############################################################
# Compiled kernel (bytecode module → loadable handle)
# ############################################################

type
  CompiledKernel* = ref object
    module*: BytecodeModule
    gpuArch*: string
    handle*: KernelHandle
    funcName*: string
    cacheDir*: string

proc compileKernel*(
    m: BytecodeModule,
    funcName: string,
    gpuArch: string = "sm_120",
    cacheDir: string = "/tmp/cutile_cache"
  ): CompiledKernel =
  ## Compile a bytecode module and load the kernel.
  ##
  ## Uses the optimal strategy:
  ##   1. Direct cuModuleLoadData (CUDA 13.1+ JIT compiles TileIR)
  ##   2. tileiras → cubin → cuModuleLoad (fallback)
  result = CompiledKernel(
    module: m,
    gpuArch: gpuArch,
    funcName: funcName,
    cacheDir: cacheDir
  )
  let cudaMod = compileBytecodeCached(m, gpuArch, cacheDir)
  result.handle = loadKernelFromModule(cudaMod, funcName)

proc launch*(k: CompiledKernel, gridX: uint32 = 1,
             gridY: uint32 = 1, gridZ: uint32 = 1,
             args: openArray[pointer] = @[]) =
  launchKernel(k.handle.function, gridX, gridY, gridZ, args)

proc unload*(k: var CompiledKernel) =
  if k.handle != nil:
    unload(k.handle)
