## cutile/examples/e01_hello_world.nim
## Port of cutile-rs tutorial 01-hello-world
##
## Minimal tile kernel that demonstrates the basic kernel builder:
## building a function, emitting ops, and producing TileIR bytecode.
##
## In cutile-rs terms this is a "hello from tile" kernel. Here we show
## the equivalent Nim pipeline: build → bytecode → (eventually) launch.

import
  std/[os, strutils],
  ../bytecode,
  ../dsl

# ############################################################
# Kernel builder
# ############################################################

proc buildHelloKernel*(): BytecodeModule =
  ## Build a minimal kernel that retrieves the tile block ID.
  ##
  ## The TileIR kernel body:
  ##   let bid = getTileBlockId()
  ##   (the scalar tile-block-id is available as a value)
  ##
  ## Inputs: none (self-contained)
  let kb = newKernel("hello_kernel", @[], @[])

  # getTileBlockId() returns the 3D tile-block coordinate as a
  # scalar I32 value.  Each launched tile gets its own ID.
  let bid = kb.getTileBlockId()
  discard bid  # available for downstream ops

  # For a real kernel this value would be stored to memory or
  # used to index into tensors.  Here we just show the pattern.
  kb.ret()
  return kb.build()

# ############################################################
# Bytecode verification
# ############################################################

proc verifyHelloBytecode*(m: BytecodeModule) =
  let bc = toBytecode(m)
  doAssert bc.len > 0, "Bytecode must not be empty"
  doAssert bc[0] == 0x7F'u8, "Expected TileIR magic byte"
  echo "  ✓ Bytecode: ", bc.len, " bytes"

  doAssert m.functions.len == 1
  doAssert m.functions[0].name == "hello_kernel"
  doAssert m.functions[0].body.len >= 1
  doAssert m.functions[0].body[0].opcode == OpGetTileBlockId
  echo "  ✓ Kernel 'hello_kernel' has ", m.functions[0].body.len, " ops"

# ############################################################
# Host runner
# ############################################################

proc runHelloWorld*() =
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  e01: Hello World (Tile Block ID)                   ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""

  let m = buildHelloKernel()
  verifyHelloBytecode(m)

  let bc = toBytecode(m)
  let tmp = "/tmp/cutile_examples"
  if not dirExists(tmp): createDir(tmp)
  let path = tmp / "e01_hello_world.bc"
  var s = newStringOfCap(bc.len)
  for b in bc: s.add(chr(b))
  writeFile(path, s)
  echo "  Wrote bytecode to: ", path
  echo ""
  echo "  To compile and launch:"
  echo "    tileiras --gpu-name sm_120 -o e01_hello_world.cubin e01_hello_world.bc"
  echo "    cuModuleLoad + cuModuleGetFunction + cuLaunchKernel grid=(n,1,1)"
  echo ""
  echo "✓ e01 hello world done"

when isMainModule:
  runHelloWorld()
