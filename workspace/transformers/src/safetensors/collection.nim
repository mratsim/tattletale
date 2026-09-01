# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Safetensors collection: a tensor-keyed view over one checkpoint.
##
## This is the storage seam between checkpoint bytes and tensor consumers:
## it resolves `model.safetensors.index.json` weight maps across shards,
## opens single-file checkpoints, and serves headers and tensors by name.
## It stays byte-to-tensor only. Config schema reads and model-level key
## semantics (prefix tallies, untied-head decisions) live with the models.
##
## Lifecycle:
## 1. `openSafetensorsCollection` parses the index of a sharded checkpoint,
##    or the header of a single-file checkpoint, then opens every shard
##    it references. Cost is the index parse plus per-shard header parse:
##    mmap is lazy, weight bytes page in on requested tensor reads only.
## 2. Header queries (shape, dtype, `shardFile`, `tensorKeys`) never
##    touch tensor bytes. `getTensor` materializes an owned copy routed
##    to any device, safe to use after close.
## 3. `close` releases every opened file. The collection then holds no
##    live views.
##
## Variant stack:
## `addVariant` attaches a second collection behind a star-glob filter
## list. Keys matching a filter resolve against the most recently added
## variant that matches, else the base collection. It mirrors
## exllamav3's `VariantSafetensorsCollection` lookup, later-added variants
## having
## priority. A collection attaches to at most one parent: a second
## `addVariant` attempt raises, so teardown can never close a variant
## attached from two places.

import
  std/algorithm,
  std/memfiles,
  std/os,
  std/sets,
  std/strutils,
  std/tables,
  pkg/packedjson,
  workspace/libtorch,
  workspace/safetensors

# #######################################################################
#
#              Safetensors collection over a checkpoint
#
# #######################################################################

const DefaultIndexName* = "model.safetensors.index.json"

type
  OpenShard = tuple
    ## One opened safetensor file. The memfile stays open for the collection
    ## lifetime: the Safetensor holds zero-copy views into the file.
    memFile: MemFile
    st: Safetensor

  VariantRule = tuple
    ## One variant lookup rule. The star-glob patterns match tensor keys
    ## served by the attached collection.
    patterns: seq[string]
    collection: SafetensorsCollection

  SafetensorsCollection* = ref object
    ## Tensor-keyed view over one checkpoint directory or file.
    ##
    ## All referenced shards stay open for the collection lifetime. Header
    ## queries cost table lookups, tensor loads materialize owned copies.
    directory*: string         ## Checkpoint directory. For a single-file open,
                               ## the file's parent directory
    indexFilename*: string     ## Index filename when opened from an index
                               ## directory, `""` otherwise
    totalSize*: int            ## `metadata.total_size` of the checkpoint
                               ## index. On an index-less file open
                               ## `summedTensorBytes` mirrors this value.
    summedTensorBytes*: int    ## Element count times byte width, summed over
                               ## all visible tensors
    weightMap: Table[string, string]  ## Tensor key -> shard filename, relative
                                      ## to `directory`
    shards: Table[string, OpenShard]  ## Shard filename -> opened shard
    variants: seq[VariantRule]        ## Variant lookup rules, most recently
                                      ## added first
    attached: bool                    ## True once bound into a parent's
                                      ## variant stack, `addVariant` refuses
                                      ## another attach

func elementCount(shape: seq[int]): int =
  ## Number of elements of a tensor shape.
  result = 1
  for dim in shape:
    result *= dim

func stElementBytes(dtype: ST_dtype): int =
  ## Byte width of one safetensors element.
  case dtype
  of BOOL, F4, F6_E2M3, F6_E3M2, U8, I8, F8_E5M2, F8_E4M3, F8_E8M0: 1
  of I16, U16, F16, BF16: 2
  of I32, U32, F32: 4
  of C64, F64, I64, U64: 8

proc requireTensor(view: SafetensorsCollection, tensorName: string) =
  ## Raises ValueError naming the tensor when the collection holds
  ## no entry for it.
  if not view.weightMap.hasKey(tensorName):
    let place = if view.indexFilename.len > 0:
      "the checkpoint index " & view.indexFilename
    else:
      "the safetensor files of " & view.directory
    raise newException(ValueError,
      "[ttt] SafetensorsCollection: tensor '" & tensorName &
      "' is absent from " & place)

proc globMatch*(key, pattern: string): bool =
  ## Star-glob full match over a tensor key, matching the variant filters
  ## of exllamav3 compiled with `compile_star_globs`: `*` matches any run
  ## of characters, everything else is literal, and the whole
  ## key must match.
  ##
  ## Example: `globMatch("layers.3.norm.weight", "layers.*.norm.*")` returns true.
  ##
  ## Reachable-positions scan: `reach[p]` holds true when the pattern prefix
  ## processed so far can consume the key up to position p, so the star
  ## ahead of a segment absorbs runs of any length. Pinning each segment
  ## at its earliest occurrence, the one-pass shortcut, misses sound
  ## matches: it places the final segment directly after the earliest
  ## occurrence of its preceding segment, so `a*bc` no longer matches
  ## `abcabc` in full.
  let segments = pattern.split('*')
  if segments.len == 1:
    return key == pattern
  var reach = newSeq[bool](key.len + 1)
  if key.startsWith(segments[0]):
    reach[segments[0].len] = true
  for i in 1 ..< segments.len - 1:
    let seg = segments[i]
    if seg.len == 0:
      # Adjacent stars match any run, reachability is preserved.
      continue
    var next = newSeq[bool](key.len + 1)
    var reached = false
    for q in 0 .. key.len - seg.len:
      if reach[q]:
        reached = true
      if reached and key.continuesWith(seg, q):
        next[q + seg.len] = true
    reach = next
  let last = segments[^1]
  if last.len == 0:
    # A trailing star matches a run of any length, including the empty run.
    for position in 0 .. key.len:
      if reach[position]:
        return true
    return false
  let start = key.len - last.len
  if start < 0:
    return false
  var reached = false
  for q in 0 .. start:
    if reach[q]:
      reached = true
  result = reached and key.continuesWith(last, start)

proc close*(view: SafetensorsCollection) =
  ## Close every opened shard, base collection and variants. Tensors
  ## obtained through getTensor are owned copies, safe to use after close.
  ## The collections hold no open files afterwards, and requests after close
  ## raise the loud absent-key error instead of reaching a dead view.
  if view.variants.len > 0:
    for rule in view.variants.items():
      close(rule.collection)
    view.variants.setLen(0)
  for shardName, shard in view.shards.mpairs():
    close(shard.memFile)
  view.shards.clear()
  view.weightMap.clear()

proc openShard(directory, shardName: string): OpenShard =
  ## Open one safetensor file of the checkpoint. The memfile must stay open:
  ## the Safetensor holds zero-copy views into the file. Any shard defect
  ## (absent file, malformed header) raises `ValueError` naming the file,
  ## and the raw reader's own exceptions are re-raised this way. A rejected
  ## header releases its memfile before the raise.
  let shardPath = directory / shardName
  if not fileExists(shardPath):
    raise newException(ValueError,
      "[ttt] SafetensorsCollection: shard file " & shardPath &
      " is missing")
  var opened = false
  try:
    result.memFile = memFiles.open(shardPath, mode = fmRead)
    opened = true
    result.st = safetensors.load(result.memFile)
  except Exception as err:
    # The memfile stays mapped past the open call, so a rejected header
    # must release it: no refused shard leaves a mapping behind.
    if opened:
      close(result.memFile)
    raise newException(ValueError,
      "[ttt] SafetensorsCollection: shard file " & shardPath &
      " is not a valid safetensors file: " & err.msg)

proc sumTensorBytes(view: SafetensorsCollection): int =
  ## Sum of element count times byte width over all visible tensors:
  ## the base collection, read from the shard headers only.
  for tensorName in view.weightMap.keys():
    let info = view.shards[view.weightMap[tensorName]].st.tensors[tensorName]
    result += info.shape.elementCount() * stElementBytes(info.dtype)

proc openSafetensorsCollection*(
    modelDir: string,
    indexName: string = DefaultIndexName
  ): SafetensorsCollection =
  ## Open one checkpoint as a tensor-keyed collection and validate it.
  ##
  ## `modelDir` accepts three shapes:
  ## - a checkpoint directory carrying `indexName` (default `model.safetensors.index.json`):
  ##   the weight map resolves tensor keys across the referenced shard files
  ## - a checkpoint directory without an index: every `.safetensors` file
  ##   in it is indexed, a key claimed by two files raises loudly
  ## - a `.safetensors` file path whose own header is the whole weight
  ##   map
  ##
  ## Raises ValueError naming the path for a missing or malformed index,
  ## a missing referenced shard file, a `metadata.total_size` disagreement
  ## with the summed tensor bytes, an index entry absent from its shard,
  ## a shard with a defective header, and a key claimed by two shard
  ## files. The raw reader validates every safetensor header at open
  ## time, so header defects surface here. A refused open releases
  ## the shard files it already opened, no mapping survives the raise.
  if fileExists(modelDir) and not dirExists(modelDir):
    result = SafetensorsCollection(
      directory: modelDir.parentDir(),
      shards: initTable[string, OpenShard]()
    )
    let shardName = modelDir.lastPathPart
    result.shards[shardName] = openShard(result.directory, shardName)
    for tensorName in result.shards[shardName].st.tensors.keys():
      result.weightMap[tensorName] = shardName
    result.summedTensorBytes = sumTensorBytes(result)
    result.totalSize = result.summedTensorBytes
    return

  let indexPath = modelDir / indexName
  if fileExists(indexPath):
    var index: JsonNode
    try:
      index = indexPath.parseFile()
    except CatchableError as err:
      raise newException(ValueError,
        "[ttt] SafetensorsCollection: checkpoint index " & indexPath &
        " is not valid JSON: " & err.msg)

    let metaNode = index{"metadata"}
    if metaNode.kind != JObject:
      raise newException(ValueError,
        "[ttt] SafetensorsCollection: checkpoint index " & indexPath &
        " carries no metadata object")
    let totalSizeNode = metaNode{"total_size"}
    if totalSizeNode.kind != JInt and totalSizeNode.kind != JFloat:
      raise newException(ValueError,
        "[ttt] SafetensorsCollection: checkpoint index " & indexPath &
        " carries no numeric metadata.total_size")
    var totalSize = 0
    if totalSizeNode.kind == JInt:
      totalSize = totalSizeNode.getBiggestInt().int
    else:
      # Some checkpoints serialize total_size as a float. Accept an integral
      # one, reject a fractional one.
      let asFloat = totalSizeNode.getFloat()
      totalSize = asFloat.int
      if asFloat != totalSize.float:
        raise newException(ValueError,
          "[ttt] SafetensorsCollection: checkpoint index " & indexPath &
          " carries a fractional metadata.total_size " & $asFloat)
    let weightMapNode = index{"weight_map"}
    if weightMapNode.kind != JObject:
      raise newException(ValueError,
        "[ttt] SafetensorsCollection: checkpoint index " & indexPath &
        " carries no weight_map object")

    result = SafetensorsCollection(
      directory: modelDir,
      indexFilename: indexName,
      totalSize: totalSize,
      weightMap: initTable[string, string]()
    )
    for tensorName, shardNode in weightMapNode.pairs():
      if shardNode.kind != JString:
        raise newException(ValueError,
          "[ttt] SafetensorsCollection: weight_map entry '" & tensorName &
          "' is not a shard filename")
      result.weightMap[tensorName] = shardNode.getStr()

    var validated = false
    try:
      for shardName in result.weightMap.values():
        if shardName notin result.shards:
          result.shards[shardName] = openShard(result.directory, shardName)

      for tensorName, shardName in result.weightMap.pairs():
        let shard = result.shards[shardName]
        if not shard.st.tensors.hasKey(tensorName):
          raise newException(ValueError,
            "[ttt] SafetensorsCollection: index entry '" & tensorName &
            "' is absent from shard " & shardName)

      result.summedTensorBytes = result.sumTensorBytes()
      if result.summedTensorBytes != result.totalSize:
        raise newException(ValueError,
          "[ttt] SafetensorsCollection: summed tensor bytes " &
          $result.summedTensorBytes &
          " disagree with metadata.total_size " & $result.totalSize &
          " in " & indexPath)
      validated = true
    finally:
      # Every defect past the first opened shard releases the collection
      # again: a refused open must leave no shard mapping behind.
      if not validated:
        close(result)
    return

  # No index in the directory: index every .safetensors file it carries.
  # Sorted for a deterministic duplicate-key report.
  var shardNames: seq[string]
  for entry in walkDir(modelDir):
    let name = extractFilename(entry.path)
    if name.endsWith(".safetensors"):
      shardNames.add name
  shardNames.sort()
  if shardNames.len == 0:
    raise newException(ValueError,
      "[ttt] SafetensorsCollection: " & indexPath & " is missing and " &
      "directory " & modelDir & " carries no .safetensors file")

  result = SafetensorsCollection(
    directory: modelDir,
    weightMap: initTable[string, string]()
  )
  var indexed = false
  try:
    for shardName in shardNames:
      result.shards[shardName] = openShard(result.directory, shardName)
      for tensorName in result.shards[shardName].st.tensors.keys():
        if result.weightMap.hasKey(tensorName):
          raise newException(ValueError,
            "[ttt] SafetensorsCollection: tensor '" & tensorName &
            "' is claimed by the shard files " & result.weightMap[tensorName] &
            " and " & shardName)
        result.weightMap[tensorName] = shardName
    result.summedTensorBytes = result.sumTensorBytes()
    result.totalSize = result.summedTensorBytes
    indexed = true
  finally:
    # Same refusal contract as the index path: a refused indexing pass
    # releases every shard file it opened along the way.
    if not indexed:
      close(result)

proc addVariant*(
    view: SafetensorsCollection,
    patterns: seq[string],
    variant: SafetensorsCollection
  ) =
  ## Attach a variant collection behind star-glob filters, for example
  ## `["model.language_model.layers.*.norm.*", "lm_head.weight"]`.
  ## Lookup walks the most recently added variants first: a later add
  ## shadows an earlier one for the keys both match. The variant keeps
  ## its own open shards and is closed together with `view`.
  ##
  ## Attach-once: each collection serves as a variant of at most one
  ## parent. A second attach raises `ValueError`, because teardown closes
  ## the variant with its parent and a double attach would close it twice.
  if variant == view:
    raise newException(ValueError,
      "[ttt] SafetensorsCollection: a collection cannot be its own variant")
  if variant.attached:
    raise newException(ValueError,
      "[ttt] SafetensorsCollection: a collection cannot be attached as a " &
      "variant twice")
  variant.attached = true
  view.variants.insert((patterns: patterns, collection: variant), 0)

proc findCollection(view: SafetensorsCollection, tensorName: string): SafetensorsCollection =
  ## The collection that serves a tensor name under the variant stack.
  result = view
  for rule in view.variants:
    for pattern in rule.patterns:
      if globMatch(tensorName, pattern):
        return rule.collection

proc getTensor*(
    view: SafetensorsCollection,
    tensorName: string,
    device = kCPU
  ): Tensor =
  ## Owned copy of one tensor, routed to `device`, rank-agnostic.
  ## Lookup runs through the variant stack first, then the checkpoint
  ## weight map. Raises ValueError naming the tensor when no serving
  ## collection has an entry for it.
  let target = view.findCollection(tensorName)
  if target != view:
    return target.getTensor(tensorName, device)
  requireTensor(view, tensorName)
  view.shards[view.weightMap[tensorName]].st.getTensorOwned(tensorName, device)

proc hasTensor*(view: SafetensorsCollection, tensorName: string): bool =
  ## Key presence under the variant stack: true when the serving collection
  ## has an entry for tensorName.
  let target = view.findCollection(tensorName)
  if target != view:
    return target.hasTensor(tensorName)
  view.weightMap.hasKey(tensorName)

proc variantStack(view: SafetensorsCollection): seq[SafetensorsCollection] =
  ## The variant collections of `view` in lookup order, depth first:
  ## each variant is listed before its own variants, and a variant
  ## added recently ranks before one added earlier. A shared collection
  ## cannot occur, `addVariant` attaches once.
  var stack: seq[SafetensorsCollection]
  for i in countdown(view.variants.high, 0):
    stack.add view.variants[i].collection
  while stack.len > 0:
    let coll = stack.pop()
    result.add coll
    for i in countdown(coll.variants.high, 0):
      stack.add coll.variants[i].collection

iterator tensorKeys*(view: SafetensorsCollection): string =
  ## Every visible tensor key, in shard order: the keys of the base
  ## collection, then the variant-only keys. Variant-only keys follow
  ## the lookup order of `variantStack`. A key served by the base
  ## and by a matching variant alike is yielded once, from the base.
  var seen = initHashSet[string]()
  for tensorName in view.weightMap.keys():
    seen.incl tensorName
    yield tensorName
  for coll in variantStack(view):
    for tensorName in coll.weightMap.keys():
      if tensorName notin seen:
        seen.incl tensorName
        yield tensorName

proc tensorCount*(view: SafetensorsCollection): int =
  ## Number of distinct visible tensor keys: the base collection's keys
  ## plus the variant-only keys. A key served by base and a matching
  ## variant alike counts once.
  var seen = initHashSet[string]()
  for tensorName in view.weightMap.keys():
    seen.incl tensorName
  result = view.weightMap.len
  for coll in variantStack(view):
    for tensorName in coll.weightMap.keys():
      if tensorName notin seen:
        seen.incl tensorName
        inc result


proc tensorInfo*(view: SafetensorsCollection, tensorName: string): TensorInfo =
  ## Header info of tensorName as served under the variant stack, no tensor
  ## load. Raises ValueError naming the tensor when absent.
  let target = view.findCollection(tensorName)
  if target != view:
    return target.tensorInfo(tensorName)
  requireTensor(view, tensorName)
  view.shards[view.weightMap[tensorName]].st.tensors[tensorName]

proc shardFile*(view: SafetensorsCollection, tensorName: string): string =
  ## Shard filename the weight map assigns to tensorName, verified present
  ## under the variant stack. On an index-less open the returned
  ## filename is the file's base name relative to `directory`.
  discard view.tensorInfo(tensorName)
  let target = view.findCollection(tensorName)
  if target != view:
    return target.weightMap[tensorName]
  view.weightMap[tensorName]

proc tensorShape*(view: SafetensorsCollection, tensorName: string): seq[int] =
  ## Shape of tensorName from the served shard header, no tensor load.
  view.tensorInfo(tensorName).shape

proc tensorDtype*(view: SafetensorsCollection, tensorName: string): ST_dtype =
  ## Safetensors dtype of tensorName from the served shard header, no tensor
  ## load.
  view.tensorInfo(tensorName).dtype
