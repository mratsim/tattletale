# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off
##   --outdir:build/wip --nimcache:nimcache/wip
##   workspace/transformers/tests/test_safetensors_collections.nim

## Checks on the safetensors collection opened from a checkpoint index.
## The index is untrusted input. Every `weight_map` value names a safetensor
## file that sits directly inside the checkpoint directory.
## Each falsifier writes an index in a temp directory, expecting the open
## to raise an IOError that names both tensor and offending filename.

import
  std/algorithm,
  std/importutils,
  std/os,
  std/strutils,
  std/tables,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/libtorch_testutils

# Routing and header questions the tests answer by peeking the private tables.
import workspace/safetensors/src/safetensors {.all.}
import workspace/safetensors/src/collections {.all.}
privateAccess(SafetensorObj)
privateAccess(SafetensorsCollectionObj)

const FixturesDir = currentSourcePath().parentDir() / "fixtures"
let HadamardFixture = FixturesDir / "exl3-hadamard" / "hadamard_single_block.safetensor"
let NormFixture = FixturesDir / "layers" / "Qwen3-0.6B-layer-8" / "norm-Qwen3-0.6B-01.safetensor"

proc tensorNames(path: string): seq[string] =
  ## Sorted tensor keys of one safetensor file, read from its header.
  let st = Safetensor.open(path)
  for tensorName in st.tensors.keys():
    result.add tensorName
  result.sort()

proc tensorBytesSum(path: string): int =
  ## Element count times byte width summed over one fixture's tensors.
  ## Covers the dtypes the fixtures use and rejects any other.
  let st = Safetensor.open(path)
  for tensorName, info in st.tensors:
    case info.dtype
    of F16, BF16:
      var numel = 1
      for dim in info.shape:
        numel *= dim
      result += numel * 2
    else:
      doAssert false, "unexpected fixture dtype " & $info.dtype

func indexJson(totalSize: int, entries: seq[(string, string)]): string =
  ## Minimal checkpoint index: metadata.total_size plus a weight_map
  ## carrying the given tensor-key to filename pairs.
  result = "{\"metadata\":{\"total_size\":" & $totalSize & "},\"weight_map\":{"
  for index, (tensorName, fileName) in entries.pairs():
    if index > 0:
      result.add ","
    result.add "\"" & tensorName & "\":\"" & fileName.replace("\\", "\\\\") & "\""
  result.add "}}"

proc hadamardEntries(fileAs: string): seq[(string, string)] =
  ## Weight-map entries mapping every hadamard fixture tensor to `fileAs`.
  for tensorName in tensorNames(HadamardFixture):
    result.add (tensorName, fileAs)

proc normEntries(fileAs: string): seq[(string, string)] =
  ## Weight-map entries mapping every norm fixture tensor to `fileAs`.
  for tensorName in tensorNames(NormFixture):
    result.add (tensorName, fileAs)

proc writeShardCopies(ckpt: string) =
  ## Copy both fixtures into the checkpoint directory under the sharded
  ## names a real multi-file checkpoint uses.
  createDir(ckpt)
  (ckpt / "model-00001-of-00002.safetensors").writeFile(HadamardFixture.readFile())
  (ckpt / "model-00002-of-00002.safetensors").writeFile(NormFixture.readFile())

proc expectRefused(ckpt, badTensor, badFilename: string) =
  ## Opening the checkpoint must raise an IOError naming the offending
  ## tensor entry and the filename it carries.
  var raised = false
  try:
    discard SafetensorsCollection.open(ckpt)
  except IOError as err:
    raised = true
    doAssert err.msg.startsWith "safetensors: collection:",
      "unexpected error text: " & err.msg
    doAssert badTensor in err.msg,
      "the raise must name the tensor: " & err.msg
    doAssert badFilename in err.msg,
      "the raise must name the offending filename: " & err.msg
  except ValueError as err:
    doAssert false,
      "a rejected path must raise IOError, not ValueError: " & err.msg
  doAssert raised,
    "the index entry '" & badFilename & "' was accepted"

proc main() =
  let root = getTempDir() / ("tt-safetensors-collection-" & $getCurrentProcessId())
  removeDir(root)
  createDir(root)
  defer: removeDir(root, checkDir = true)
  let hadamardBytes = tensorBytesSum(HadamardFixture)
  let normBytes = tensorBytesSum(NormFixture)
  let bothFixturesBytes = hadamardBytes + normBytes
  let absoluteDecoy = root / "abs-decoy.safetensors"
  absoluteDecoy.writeFile(NormFixture.readFile())

  runCppTest "A crafted index cannot name files outside the checkpoint directory":
    proc(): bool =
      # Each case names one weight_map value a direct-child checkpoint never
      # legitimately carries. The offending entries map the norm fixture
      # tensors. Where noted, the target really exists, so a missing
      # validation would open it instead of raising.
      let badNames = @[
        "../../evil.safetensors",  # parent traversal
        "../evil.safetensors",     # parent traversal, decoy one level up
        "",                        # the empty string names no file
        "..",                      # the parent directory itself
        ".",                       # the checkpoint directory itself
        "./x.safetensors",         # explicit current directory, decoy inside
        absoluteDecoy,             # absolute path, decoy exists
        "\\tmp\\tt-backslash.safetensors",
        "..\\evil.safetensors",    # Windows parent traversal
        "C:\\evil.safetensors",    # Windows absolute drive path
        "C:evil.safetensors"       # Windows drive-relative path
      ]
      for caseIndex, bad in badNames:
        let caseDir = root / ("refuse-" & $caseIndex)
        let ckpt = caseDir / "ckpt"
        defer: removeDir(caseDir, checkDir = true)
        var entries = hadamardEntries("model-00001-of-00002.safetensors")
        var totalSize = bothFixturesBytes
        if bad == "./x.safetensors":
          # Only the x names are referenced, and x resolves to a real
          # copy of the norm fixture directly inside the checkpoint.
          entries = normEntries(bad)
          totalSize = normBytes
        else:
          entries.add normEntries(bad)
        writeShardCopies(ckpt)
        case bad
        of "../evil.safetensors":
          (caseDir / "evil.safetensors").writeFile(NormFixture.readFile())
        of "./x.safetensors":
          (ckpt / "x.safetensors").writeFile(NormFixture.readFile())
        else:
          discard
        (ckpt / DefaultIndexName).writeFile(indexJson(totalSize, entries))
        expectRefused(ckpt, "input_hidden_states", bad)
      true

  runCppTest "Legitimate sharded and dotfile filenames open":
    proc(): bool =
      let shardDir = root / "shards"
      writeShardCopies(shardDir)
      (shardDir / DefaultIndexName).writeFile(indexJson(
        bothFixturesBytes,
        hadamardEntries("model-00001-of-00002.safetensors") &
        normEntries("model-00002-of-00002.safetensors")))
      let view = SafetensorsCollection.open(shardDir)
      doAssert view.weightMap.len == 9
      doAssert view.weightMap["svh"] == "model-00001-of-00002.safetensors"
      doAssert view.weightMap["output"] == "model-00002-of-00002.safetensors"
      doAssert view.hasTensor("input_hidden_states")
      let tensor = view.getTensorOwned("svh")
      doAssert tensor.shape == @[128]

      # A dot-prefixed filename is a legal direct child, over-rejection
      # of dotfiles would fail here.
      let dotDir = root / "dot"
      createDir(dotDir)
      (dotDir / "..hidden.safetensors").writeFile(HadamardFixture.readFile())
      (dotDir / DefaultIndexName).writeFile(
        indexJson(hadamardBytes, hadamardEntries("..hidden.safetensors")))
      let dotView = SafetensorsCollection.open(dotDir)
      doAssert dotView.weightMap.len == 7
      true

  runCppTest "A single-file open is untouched by the index validation":
    proc(): bool =
      let singleDir = root / "single"
      createDir(singleDir)
      let file = singleDir / "weights.safetensors"
      file.writeFile(HadamardFixture.readFile())
      let view = SafetensorsCollection.open(file)
      doAssert view.weightMap.len == 7
      true

main()
