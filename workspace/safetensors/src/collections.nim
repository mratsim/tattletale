# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Safetensors collection: a tensor-keyed index over one checkpoint.
##
## Resolves the `model.safetensors.index.json` weight map across the checkpoint's safetensor files.
## Serves tensor bytes by name, routed to the file that owns each key.
## Single-file checkpoints open through the same entry point.
## The safetensor files are the source of truth for the data they contain,
## the collection mirrors no dtype, shape or byte of its own.
## Byte-to-tensor only.
## Config schema reads and model-level key semantics such as prefix tallies and untied-head decisions live with the models.
##
## Lifecycle:
## 1. `SafetensorsCollection.open` parses the index of a multi-file checkpoint,
##    or the header of a single-file checkpoint, and opens every file it references.
##    The cost is the index parse and the per-file header parse.
##    Each file's mmap is lazy.
##    Weight bytes page in on requested tensor reads only.
## 2. `getMmapView` serves a zero-copy view from the file that owns the key.
##    Owned copies arrive through the libtorch bridge in `safetensors_libtorch`.
## 3. The last reference to the collection releases every opened file.

import
  std/algorithm,
  std/memfiles,
  std/os,
  std/strutils,
  std/tables,
  std/importutils,
  pkg/packedjson

import ./safetensors {.all.}
privateAccess(SafetensorObj)

# #######################################################################
#
#              Safetensors collection over a checkpoint
#
# #######################################################################

const DefaultIndexName* = "model.safetensors.index.json"

template checkValue(cond: bool; msg: string) =
  ## Raises `ValueError` with `msg` when `cond` is false.
  ## The `ValueError` class marks a defect in the checkpoint content.
  ## Internal invariants stay on `doAssert`.
  if not cond:
    raise newException(ValueError, msg)

template checkFilePath(cond: bool; msg: string) =
  ## Raises `IOError` with `msg` when `cond` is false.
  ## The `IOError` class marks a rejected name as a path defect.
  if not cond:
    raise newException(IOError, msg)

type
  SafetensorsCollection* = ref SafetensorsCollectionObj
    ## Tensor-keyed index over one checkpoint directory or file.

  SafetensorsCollectionObj = object
    weightMap: Table[string, string] ## Tensor key to safetensor filename, relative to the checkpoint directory.
    files: Table[string, Safetensor] ## Filename -> loaded safetensor file

func elementCount(shape: seq[int]): int =
  result = 1
  for dim in shape:
    result *= dim

func stElementBytes(dtype: ST_dtype): int =
  case dtype
  of BOOL, F4, F6_E2M3, F6_E3M2, U8, I8, F8_E5M2, F8_E4M3, F8_E8M0: 1
  of I16, U16, F16, BF16: 2
  of I32, U32, F32: 4
  of C64, F64, I64, U64: 8

proc isDirectChildPath(filename: string): bool =
  ## True when `filename` names a file directly inside the opened directory.
  #
  # Names with a separator are rejected outright.
  # A surviving value is a single segment, so only the two equalities below can name a directory.
  # Dot-prefixed names such as `..hidden.safetensors` are ordinary files and stay accepted.
  if filename.len == 0:
    return false
  if '/' in filename or '\\' in filename:
    return false
  if filename.len >= 2 and filename[1] == ':' and filename[0].isAlphaAscii:
    return false
  if filename == "." or filename == "..":
    return false
  true

func assignedFile(view: SafetensorsCollection, tensorName: string): string =
  ## The safetensor filename that the weight map assigns to `tensorName`.
  ## Raises ValueError naming the tensor when the collection holds no entry.
  if not view.weightMap.hasKey(tensorName):
    raise newException(ValueError,
      "safetensors: collection: tensor '" & tensorName &
      "' is absent from the checkpoint")
  view.weightMap[tensorName]

proc getMmapView*(view: SafetensorsCollection, tensorName: string): MemSlice {.inline.} =
  ## Zero-copy `MemSlice` view of the tensor data of `tensorName`, served
  ## by the file that owns the key.
  ##
  ## Preconditions: `view` is a collection returned by `open` and still alive,
  ## `tensorName` is one of its tensor keys.
  ## Postconditions: the view borrows the owning file's mapping and follows
  ## the same lifetime contract as `safetensors.getMmapView`.
  ## Raises ValueError naming the tensor when the collection holds no entry.
  privateAccess(SafetensorObj)
  view.files[view.assignedFile(tensorName)].getMmapView(tensorName)

proc open*(_: typedesc[SafetensorsCollection], path: string): SafetensorsCollection =
  ## Open one checkpoint as a tensor-keyed collection and validate it.
  ##
  ## `path` accepts three shapes:
  ## - a checkpoint directory carrying `model.safetensors.index.json`:
  ##   the weight map resolves tensor keys across the referenced safetensor files
  ## - a checkpoint directory without an index: every `.safetensors` file in it is indexed.
  ##   A key claimed by two files raises, naming the key.
  ## - a `.safetensors` file path whose own header is the whole weight map
  ##
  ## Raises ValueError naming the tensor or file for these defects:
  ## - an index that is missing or malformed
  ## - a referenced safetensor file that is missing
  ## - an index entry that its safetensor file does not carry
  ## - a file whose header is defective
  ## - a key claimed by two safetensor files
  ## Raises IOError naming the tensor and the filename for a weight_map value
  ## that names something other than a safetensor file directly inside `path`.
  ## This covers separators, absolute paths, drive paths, `.`, `..` and the empty string.
  ## The reader validates every safetensor header at open time.
  ## A refused open releases the safetensor files it already opened.
  var
    weightMap = initTable[string, string]()
    files = initTable[string, Safetensor]()
  privateAccess(SafetensorObj)

  if fileExists(path) and not dirExists(path):
    let fileName = path.lastPathPart
    files[fileName] = Safetensor.open(path)
    for tensorName in files[fileName].tensors.keys():
      weightMap[tensorName] = fileName
    return SafetensorsCollection(weightMap: weightMap, files: files)

  let indexPath = path / DefaultIndexName
  if fileExists(indexPath):
    let index = indexPath.parseFile()

    let metaNode = index{"metadata"}
    checkValue(metaNode.kind == JObject,
      "safetensors: collection: checkpoint index " & indexPath &
      " carries no metadata object")
    let totalSizeNode = metaNode{"total_size"}
    checkValue(totalSizeNode.kind == JInt or totalSizeNode.kind == JFloat,
      "safetensors: collection: checkpoint index " & indexPath &
      " carries no numeric metadata.total_size")
    let asFloat = totalSizeNode.getFloat()
    let totalSize = asFloat.int
    # Receipt: Qwen3.6-35B-A3B ships "total_size": 71903645408.0.
    # A writer emitting a decimal point stays accepted while the value is an integer.
    # A fractional value is rejected.
    checkValue(asFloat == totalSize.float,
      "safetensors: collection: checkpoint index " & indexPath &
      " carries a fractional metadata.total_size " & $asFloat)
    let weightMapNode = index{"weight_map"}
    checkValue(weightMapNode.kind == JObject,
      "safetensors: collection: checkpoint index " & indexPath &
      " carries no weight_map object")

    for tensorName, fileNode in weightMapNode.pairs():
      checkValue(fileNode.kind == JString,
        "safetensors: collection: weight_map entry '" & tensorName &
        "' is not a safetensor filename")
      let fileName = fileNode.getStr()
      checkFilePath(isDirectChildPath(fileName),
        "safetensors: collection: weight_map entry '" & tensorName &
        "' names '" & fileName &
        "', which must be a safetensor file directly inside the checkpoint directory")
      weightMap[tensorName] = fileName

    for fileName in weightMap.values():
      if fileName notin files:
        files[fileName] = Safetensor.open(path / fileName)

    for tensorName, fileName in weightMap.pairs():
      checkValue(files[fileName].tensors.hasKey(tensorName),
        "safetensors: collection: index entry '" & tensorName &
        "' is absent from the safetensor file " & fileName)

    # metadata.total_size is recorded, not enforced: real checkpoints
    # ship stale counts. Tensor integrity is enforced at open time
    # instead (the header parses, every weight_map entry exists).
    return SafetensorsCollection(weightMap: weightMap, files: files)

  # No index in the directory: index every .safetensors file it carries.
  # Sorted for a deterministic duplicate-key report.
  var fileNames: seq[string]
  for entry in walkDir(path):
    let name = extractFilename(entry.path)
    if name.endsWith(".safetensors"):
      fileNames.add name
  fileNames.sort()
  checkValue(fileNames.len != 0,
    "safetensors: collection: " & indexPath & " is missing and " &
    "directory " & path & " carries no .safetensors file")

  for fileName in fileNames:
    files[fileName] = Safetensor.open(path / fileName)
    for tensorName in files[fileName].tensors.keys():
      checkValue(not weightMap.hasKey(tensorName),
        "safetensors: collection: tensor '" & tensorName &
        "' is claimed by the safetensor files " & weightMap[tensorName] &
        " and " & fileName)
      weightMap[tensorName] = fileName

  SafetensorsCollection(weightMap: weightMap, files: files)
