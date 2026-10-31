## HLSL test utilities:
## - DXC compiles emitted kernel source twice (compileToSpirV, compileCheck):
##   DXIL output, then SPIR-V through spirv-val
## - SPIR-V-only route for the fvVulkan flavor (compileFvVulkanSpirV):
##   dxc -spirv with no DXIL stage, since the flavor emits `vk::` constructs
##   that the DXIL backend rejects
## - SPIR-V binding reflection for the descriptor-route tests (assertHlslBindings)
##
## Shared by the tests under `tests/codegen/vulkan_via_hlsl/`,
## imported as `import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils`.

import std/[os, osproc, strutils, tables]

type BindingExpectation* = tuple[name: string, ordinal: int, readOnly: bool]

proc compileFvVulkanSpirV*(name, source: string): seq[uint32] =
  ## Compiles one fvVulkan-flavor kernel to SPIR-V through the validated route:
  ## - dxc -T cs_6_4 -E main -spirv -Fo with no DXIL output, the fvVulkan
  ##   flavor emits `vk::` push-constant and BufferPointer constructs
  ##   outside the DXIL language
  ## - spirv-val --target-env vulkan1.1
  ## Returns the SPIR-V words for vkCreateShaderModule.
  ## Artifacts go to `build/tests/hlsl/<name>.hlsl(.spv)`.
  ## A non-zero exit fails the test.
  ##
  ## -E is required, DXC has no default entry point.
  let dxc = findExe("dxc")
  doAssert dxc.len > 0, "dxc not found on PATH, the HLSL tests need it"
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0, "spirv-val not found on PATH, the HLSL tests need it"
  let dir = "build" / "tests" / "hlsl"
  createDir(dir)
  let path = dir / name & ".hlsl"
  writeFile(path, source)
  let (spvOut, spvCode) = execCmdEx(
    dxc & " -T cs_6_4 -E main -spirv -Fo " & path & ".spv " & path)
  doAssert spvCode == 0,
    "dxc SPIR-V leg failed (exit " & $spvCode & ") for " & name & ":\n" & spvOut
  let (valOut, valCode) = execCmdEx(
    spirvVal & " --target-env vulkan1.1 " & path & ".spv")
  doAssert valCode == 0,
    "spirv-val failed (exit " & $valCode & ") for " & name & ":\n" & valOut
  let raw = readFile(path & ".spv")
  result = newSeq[uint32](raw.len div 4)
  copyMem(result[0].addr, raw[0].addr, raw.len)

proc compileToSpirV*(name, source: string): seq[uint32] =
  ## Compiles one emitted kernel through the validated DXC route:
  ## - dxc -T cs_6_4 -E main -Fo, the DXIL output
  ## - dxc -T cs_6_4 -E main -spirv -Fo, the SPIR-V module
  ## - spirv-val --target-env vulkan1.1
  ## Returns the SPIR-V words for vkCreateShaderModule.
  ## Artifacts go to `build/tests/hlsl/<name>.hlsl(.cso)(.spv)`.
  ## A non-zero exit fails the test.
  ##
  ## -E is required, DXC has no default entry point.
  ## This DXC 1.11 build rejects the cs_6_x profile alias, cs_6_4 is the profile in use.
  let dxc = findExe("dxc")
  doAssert dxc.len > 0, "dxc not found on PATH, the HLSL tests need it"
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0, "spirv-val not found on PATH, the HLSL tests need it"
  let dir = "build" / "tests" / "hlsl"
  createDir(dir)
  let path = dir / name & ".hlsl"
  writeFile(path, source)
  let (dxilOut, dxilCode) = execCmdEx(
    dxc & " -T cs_6_4 -E main -Fo " & path & ".cso " & path)
  doAssert dxilCode == 0,
    "dxc DXIL compilation failed (exit " & $dxilCode & ") for " & name & ":\n" & dxilOut
  # HLSL shuffle builtins need a Vulkan target env on the SPIR-V module,
  # DXC refuses them under the default env. vulkan1.1 keeps the emitted
  # SPIR-V at version 1.3, the version spirv-val validates under.
  let (spvOut, spvCode) = execCmdEx(
    dxc & " -T cs_6_4 -E main -spirv -fspv-target-env=vulkan1.1 -Fo " & path & ".spv " & path)
  doAssert spvCode == 0,
    "dxc SPIR-V leg failed (exit " & $spvCode & ") for " & name & ":\n" & spvOut
  let (valOut, valCode) = execCmdEx(
    spirvVal & " --target-env vulkan1.1 " & path & ".spv")
  doAssert valCode == 0,
    "spirv-val failed (exit " & $valCode & ") for " & name & ":\n" & valOut
  let raw = readFile(path & ".spv")
  result = newSeq[uint32](raw.len div 4)
  copyMem(result[0].addr, raw[0].addr, raw.len)

proc assertHlslBindings*(spvPath: string, expected: openArray[BindingExpectation]) =
  ## Checks the compiled module's SPIR-V reflection against the printer's
  ## binding assignment. Per buffer resource:
  ## - the Binding decoration matches the printer's parameter-order numbering, set 0
  ## - no resource is NonWritable: DXC emits no read-only decoration,
  ##   the printer's write-scan is the only read-only source (it chooses t vs u registers)
  ##
  ## Binding-order mismatches are silent on MoltenVK with zero diagnostics,
  ## the disassembly is the guard.
  let spirvDis = findExe("spirv-dis")
  doAssert spirvDis.len > 0, "spirv-dis not found on PATH, the HLSL binding reflection needs it"
  let (disOut, disCode) = execCmdEx(spirvDis & " " & spvPath)
  doAssert disCode == 0, "spirv-dis failed on " & spvPath & ":\n" & disOut

  var idNames: Table[string, string]   # SPIR-V %id -> OpName text
  var decorations: Table[string, tuple[binding, setNum: int, nonWritable: bool]]
  for line in disOut.splitLines():
    let t = line.strip()
    if t.startsWith("OpName "):
      let rest = t["OpName ".len .. ^1]
      let sp = rest.find(' ')
      if sp > 0:
        idNames[rest[0 ..< sp]] = rest[sp + 1 .. ^1].strip(chars = {'"'})
    elif t.startsWith("OpDecorate "):
      # Decoration line: `OpDecorate %id <decoration> <value?>`.
      # Duplicate OpNames gain a `_0` compiler suffix.
      # Decorations sit on the resource variable (not on its block type).
      let parts = t.split(' ')
      if parts.len >= 3:
        case parts[2]
        of "Binding":
          decorations.mgetOrPut(parts[1], (-1, -1, false)).binding = parseInt(parts[3])
        of "DescriptorSet":
          decorations.mgetOrPut(parts[1], (-1, -1, false)).setNum = parseInt(parts[3])
        of "NonWritable":
          decorations.mgetOrPut(parts[1], (-1, -1, false)).nonWritable = true
        else:
          discard

  for (name, ordinal, readOnly) in expected:
    var found = false
    for id, nm in idNames:
      if nm != name or id notin decorations:
        continue
      doAssert decorations[id].binding == ordinal,
        "binding reflection: " & name & " sits on binding " & $decorations[id].binding &
        ", the printer numbered it " & $ordinal
      doAssert decorations[id].setNum == 0,
        "binding reflection: " & name & " sits in descriptor set " &
        $decorations[id].setNum & ", want 0"
      doAssert not decorations[id].nonWritable,
        "binding reflection: " & name & " carries NonWritable, DXC emits none"
      found = true
      break
    doAssert found,
      "binding reflection: no buffer resource named " & name & " carries a Binding decoration"
  echo "  binding reflection: ", expected.len, " buffer resources on their printer-numbered bindings, set 0"

proc compileCheck*(name, source: string, expected: openArray[BindingExpectation]): seq[uint32] =
  ## Compiles one emitted kernel through the validated route.
  ## Returns the SPIR-V words for vkCreateShaderModule.
  ##
  ## - buffer resources in `expected` trigger the binding reflection
  ##   (assertHlslBindings) over the compiled module
  ## - an empty `expected` is the compile-only path for kernels without
  ##   buffer resources
  ## No HLSL engine exists, this compile check substitutes for `engine.launchKernel()` (house exception for backends without runtime support).
  result = compileToSpirV(name, source)
  if expected.len == 0:
    echo name, ": dxc + spirv-val ok"
  else:
    assertHlslBindings("build" / "tests" / "hlsl" / name & ".hlsl.spv", expected)
    echo name, ": dxc + spirv-val + binding reflection ok"
