# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Device selection for the transformer suites.
##
## Suites resolve the compute device through one code path,
## `testDevice()`, instead of hardcoding a device per suite.
## The compile-time string define `TTT_TEST_ON`
## (auto | metal | cpu | cuda) selects it.
## An explicit value wins over the platform default:
##
## - auto: Metal on macOS, CUDA when a Linux host provides one
##   (`Torch.cuda_is_available()`), CPU as the last resort
## - metal, cpu, cuda: the named device on any platform
##
## The per-suite test tasks translate an environment `TTT_TEST_ON`
## value into this define through config.nims. Therefore
## `TTT_TEST_ON=cpu nim test_tf_bf16_qwen3_03_chain`
## flips the device without a hand-written build command.
##
## On a GPU-less host the auto default asks
## `Torch.cuda_is_available()` at run time inside the compiled suite.
## A GPU-less Apple Silicon host always takes the macOS Metal branch, so the GPU-less
## branch is verified by the define logic alone.
##
## Device comparison: the recorded device (the fixture manifest's
## `recorded_from` row) compared against the run device (`testDevice()`)
## selects the tolerance class:
## - same device on both sides: the run computes with the recorded
##   kernels in the recorded call order, outputs compare bit-exactly
## - different devices: outputs compare under the cross-device drift
##   tolerances

import
  std/os,
  std/strutils,
  workspace/libtorch as F,
  ./provenance,
  ./tolerance

const
  TTT_TEST_ON* {.strdefine.} = "auto"
    ## Compile-time device selection: auto, metal, cpu or cuda.

proc parseTestDevice*(name: string): F.DeviceKind =
  ## The device named by one TTT_TEST_ON value. Unknown names raise.
  case name
  of "metal": F.kMPS
  of "cpu": F.kCPU
  of "cuda": F.kCUDA
  else:
    raise newException(ValueError,
      "TTT_TEST_ON must name auto, metal, cpu or cuda, got: " & name)

proc testDevice*(): F.DeviceKind =
  ## The device the suites compute on. An explicit TTT_TEST_ON value
  ## wins. The auto default is platform-dependent: Metal on macOS,
  ## CUDA when a Linux host provides one, CPU as the last resort.
  when TTT_TEST_ON == "auto":
    when defined(macosx):
      F.kMPS
    else:
      if F.Torch.cuda_is_available(): F.kCUDA else: F.kCPU
  else:
    parseTestDevice(TTT_TEST_ON)

proc deviceName*(device: F.DeviceKind): string =
  ## Stable printed name of a device kind, used by suite reports
  ## and by the device comparison lines.
  case device
  of F.kCPU: "cpu"
  of F.kCUDA: "cuda"
  of F.kMPS: "metal"
  else: $device

type
  DeviceCompareClass* = enum
    ## The tolerance class the device comparison selects.
    sameDeviceBitExact
      ## The run computes on the recorded device itself: both sides
      ## call the same kernels in the same call order, so computed
      ## outputs compare bit-exactly, any drift is a bug.
    crossDeviceDrift
      ## The run computes on a different device than the recording:
      ## computed outputs compare under the cross-device drift tolerances,
      ## the per-op tolerances and the chain checkpoint band.

proc recordedDevice*(recordedFromValue: string): F.DeviceKind =
  ## The record-time device named by a `recorded_from` manifest value.
  ## The value names box plus device, "m4max-cpu" being the shape.
  let sep = recordedFromValue.rfind('-')
  if sep < 1:
    raise newException(ValueError,
      "recorded_from must name box-device, got: " & recordedFromValue)
  case recordedFromValue[sep + 1 .. ^1]
  of "cpu": F.kCPU
  of "metal": F.kMPS
  of "cuda": F.kCUDA
  else:
    raise newException(ValueError,
      "recorded_from device must name cpu, metal or cuda, got: " &
      recordedFromValue)

proc manifestValue*(fixtureDir: string, key: string): string =
  ## One stamped PROVENANCE.md row: the manifest-declared fact the
  ## suites read. The file is verified by re-render before the read.
  let path = fixtureDir / "PROVENANCE.md"
  if not fileExists(path):
    raise newException(IOError,
      "fixture family carries no PROVENANCE.md manifest: " & fixtureDir)
  if not verifyProvenance(path):
    raise newException(ValueError,
      "PROVENANCE.md fails the byte-level re-render: " & path)
  for (k, v) in parseProvenance(readFile(path)):
    if k == key:
      return v
  raise newException(ValueError,
    "PROVENANCE.md carries no " & key & " row: " & path)

proc recordedFrom*(fixtureDir: string): string =
  ## The manifest-declared record-time device of a fixture family:
  ## its `recorded_from` PROVENANCE.md row, re-render verified
  ## before the read.
  manifestValue(fixtureDir, "recorded_from")

proc compareClass*(recorded, run: F.DeviceKind): DeviceCompareClass =
  ## The tolerance class the device comparison selects: the recorded
  ## device off the fixture manifest, the run device off `testDevice()`.
  if run == recorded:
    sameDeviceBitExact
  else:
    crossDeviceDrift

proc compareReport*(fixtureDir: string, run: F.DeviceKind): string =
  ## One-line report naming the recorded device, the run device and the
  ## tolerance class their comparison selects, the line printed per
  ## device-flipped run.
  ## The record-time device stays fixed at its manifest value, run-time
  ## follows TTT_TEST_ON.
  let rec = recordedFrom(fixtureDir)
  let row = compareClass(recordedDevice(rec), run)
  result = "recorded_from " & rec & ", run device " & deviceName(run) &
    ", tolerance class " & $row

proc compareLine*(fixtureDir: string, run: F.DeviceKind): string =
  ## The printed device-comparison line of a suite start. Fixture
  ## families without a PROVENANCE.md state no recorded device. The line
  ## names that gap: no cross-device tolerance class applies, the
  ## reference tolerances remain the only comparison class of such a
  ## family.
  let path = fixtureDir / "PROVENANCE.md"
  if not fileExists(path):
    return "no PROVENANCE.md manifest in " & fixtureDir &
      ": no recorded device stated, reference rows only, run device " &
      deviceName(run)
  compareReport(fixtureDir, run)

proc cppTorchVersion(): cstring =
  ## Raw TORCH_VERSION literal of the linked libtorch headers.
  # Assign the result rather than returning inside the emit: a raw return
  # would skip the frame epilogue the line-trace build emits around the body.
  {.emit: "result = TORCH_VERSION;".}

proc linkedTorchVersion*(): string =
  ## Version stamp of the libtorch build the suite links, e.g. "2.14.0".
  ## The suites link the venv libtorch, the same library the recordings
  ## used, so this value is what the PROVENANCE torch stamp must carry.
  $cppTorchVersion()

proc assertTorchStamp*(fixtureDir: string) =
  ## Check the environment at suite start: the linked libtorch must carry
  ## the torch version stamped in the fixture family manifest. A venv
  ## downgrade re-bases every reference row silently, so the guard fails
  ## any mismatched suite before a comparison runs.
  ## Families without manifests pass unguarded.
  let path = fixtureDir / "PROVENANCE.md"
  if not fileExists(path):
    return
  let stamped = manifestValue(fixtureDir, "torch")
  let linked = linkedTorchVersion()
  if stamped != linked:
    raise newException(HarnessCheckError,
      "linked torch " & linked & " != recorded torch " & stamped &
      " of " & path & ": the reference rows would re-base silently")
