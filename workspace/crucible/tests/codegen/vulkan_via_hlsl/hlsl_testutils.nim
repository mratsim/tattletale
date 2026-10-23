## HLSL test utilities:
## - glslang + spirv-val compile of emitted kernel source (compileToSpirV, compileCheck)
## - SPIR-V binding reflection for the descriptor-route tests (assertHlslBindings)
##
## Shared by the tests under `tests/codegen/vulkan_via_hlsl/`,
## imported as `import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils`.

import std/[os, osproc, strutils, tables]

type BindingExpectation* = tuple[name: string, ordinal: int, readOnly: bool]

proc compileToSpirV*(name, source: string): seq[uint32] =
  ## Compiles one emitted kernel through the validated route:
  ## - glslangValidator -V -S comp -D -e main
  ## - spirv-val --target-env vulkan1.1
  ## Returns the SPIR-V words for vkCreateShaderModule.
  ## Artifacts go to `build/tests/hlsl/<name>.hlsl(.spv)`.
  ## A non-zero exit fails the test.
  let glslang = findExe("glslangValidator")
  doAssert glslang.len > 0, "glslangValidator not found on PATH, the HLSL tests need it"
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0, "spirv-val not found on PATH, the HLSL tests need it"
  let dir = "build" / "tests" / "hlsl"
  createDir(dir)
  let path = dir / name & ".hlsl"
  writeFile(path, source)
  let (glslOut, glslCode) = execCmdEx(
    glslang & " -V -S comp -D -e main -o " & path & ".spv " & path)
  doAssert glslCode == 0,
    "glslangValidator failed (exit " & $glslCode & ") for " & name & ":\n" & glslOut
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
  ## - the NonWritable decoration appears exactly on read-only buffers
  ##
  ## Binding-order mismatches are silent on MoltenVK, zero diagnostics, so the disassembly is the guard.
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
      # glslang suffixes duplicate OpNames with `_0`, the resource variable (not its block type) carries the decorations
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
      doAssert decorations[id].nonWritable == readOnly,
        "binding reflection: " & name & " NonWritable is " & $decorations[id].nonWritable &
        ", want " & $readOnly
      found = true
      break
    doAssert found,
      "binding reflection: no buffer resource named " & name & " carries a Binding decoration"
  echo "  binding reflection: ", expected.len, " buffer resources on their printer-numbered bindings, set 0"

proc compileCheck*(name, source: string, expected: openArray[BindingExpectation], expect: string = ""): seq[uint32] =
  ## Compiles one emitted kernel through the validated route and returns the SPIR-V words
  ## for vkCreateShaderModule. Assertions:
  ## - `expect` (when not "") must appear in the emitted text, it locks the construct under test to the shader
  ## - when `expected` lists buffer resources, the binding reflection runs (assertHlslBindings).
  ##   An empty `expected` is the compile-only path for kernels without buffer resources
  ## No HLSL engine exists, this compile check substitutes for `engine.launchKernel()` (house exception for backends without runtime support).
  if expect.len > 0:
    doAssert source.contains(expect),
      "emitted HLSL for " & name & " is missing `" & expect & "`:\n" & source
  result = compileToSpirV(name, source)
  if expected.len == 0:
    echo name, ": glslang + spirv-val ok"
  else:
    assertHlslBindings("build" / "tests" / "hlsl" / name & ".hlsl.spv", expected)
    echo name, ": glslang + spirv-val + binding reflection ok"
