# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Selftest: the harness checks reject a seeded fault corpus and accept
## known-good platform drift. Detection floors live in the module doc
## of harness/selftest.nim. Run through `nim test_transformers` or directly:
##
## nim cpp -r -d:release --stackTrace:on --lineTrace:on --lineDir:on \
##   --outdir:build/tests/t_harness_selftest \
##   --nimcache:nimcache/tests/t_harness_selftest \
##   workspace/transformers/tests/harness/t_harness_selftest.nim

import
  std/os,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/transformers/tests/harness

proc main() =
  runCppTest "harness selftest: fault corpus rejected, drift corpus accepted":
    proc(): bool =
      runSelftest()

  runCppTest "device selection resolves the define and the platform default":
    proc(): bool =
      let dev = testDevice()
      echo "    testDevice resolves to " & deviceName(dev)
      when TTT_TEST_ON == "auto":
        when defined(macosx):
          doAssert dev == F.kMPS,
            "the auto default must resolve to Metal on macOS"
        else:
          doAssert dev in {F.kCPU, F.kCUDA},
            "the auto default must resolve to CUDA or CPU off macOS"
      else:
        doAssert dev == parseTestDevice(TTT_TEST_ON),
          "an explicit TTT_TEST_ON value must win"
      true

  runCppTest "device pairing selects tolerances from the device pair":
    proc(): bool =
      doAssert compareClass(F.kCPU, F.kCPU) == sameDeviceBitExact
      doAssert compareClass(F.kMPS, F.kMPS) == sameDeviceBitExact
      doAssert compareClass(F.kCPU, F.kMPS) == crossDeviceDrift
      doAssert compareClass(F.kCPU, F.kCUDA) == crossDeviceDrift
      doAssert recordedDevice("m4max-cpu") == F.kCPU
      doAssert recordedDevice("boxtwo-metal") == F.kMPS
      doAssert recordedDevice("rbox-cuda") == F.kCUDA
      try:
        discard recordedDevice("m4max")
        doAssert false, "recordedDevice accepted a box-only value"
      except ValueError:
        discard
      try:
        discard recordedDevice("m4max-vulkan")
        doAssert false, "recordedDevice accepted an unknown device"
      except ValueError:
        discard
      # recordedFrom reads the manifest of a real fixture family.
      let fixtureDir =
        currentSourcePath().parentDir() / ".." / "fixtures" /
        "bf16-02-first-8-layers-plus-final" / "Qwen3-0.6B"
      let rec = recordedFrom(fixtureDir)
      echo "    manifest recorded_from: " & rec
      doAssert rec == "m4max-cpu"
      let expected = if testDevice() == F.kCPU: sameDeviceBitExact
                     else: crossDeviceDrift
      doAssert compareClass(recordedDevice(rec), testDevice()) == expected
      echo "    devices: ", compareReport(fixtureDir, testDevice())
      # Every fixture family manifest verifies and names a legal device:
      # the recorded-platform facts are manifest-declared, never assumed.
      var stamped = 0
      let fixturesDir =
        currentSourcePath().parentDir() / ".." / "fixtures"
      for kind, family in walkDir(fixturesDir):
        if kind != pcDir: continue
        # The manifest sits either at the family root (the EXL3 families
        # record one checkpoint per family) or under a model directory
        # (the bf16 families record one recording per model).
        for k2, modelDir in walkDir(family):
          if k2 != pcDir: continue
          if not fileExists(modelDir / "PROVENANCE.md"): continue
          let rec = recordedFrom(modelDir)
          discard recordedDevice(rec)
          # The linked torch must match the recorded torch version. A venv
          # downgrade cannot silently desync the environment rows: the guard
          # fails the run instead.
          assertTorchStamp(modelDir)
          inc stamped
        if not fileExists(family / "PROVENANCE.md"): continue
        let rec = recordedFrom(family)
        discard recordedDevice(rec)
        assertTorchStamp(family)
        inc stamped
      echo "    fixture family manifests verified: " & $stamped
      doAssert stamped >= 8,
        "the fixture families must carry verified provenance manifests"
      # The environment guard: the linked torch against the recorded torch
      # version. A mismatched version raises an error before any comparison
      # runs.
      block:
        let badDir = getTempDir() / "ttt-torch-version-guard"
        createDir(badDir)
        writeFile(badDir / "PROVENANCE.md", renderProvenance(@[
          ("python", "3.14.1"), ("torch", "9.9.9"),
          ("transformers", "5.16.1"), ("recorded_from", "m4max-cpu")]))
        let raised = (proc(): bool =
          try:
            assertTorchStamp(badDir)
            false
          except HarnessCheckError:
            true)()
        doAssert raised, "the torch-version check accepted a mismatched version"
        echo "    linked torch ", linkedTorchVersion(),
          ", the guard rejects a mismatched torch version"
      true

when isMainModule:
  main()
