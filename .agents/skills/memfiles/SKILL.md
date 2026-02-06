---
name: memfiles
description: Memory-mapped file access for zero-copy I/O in Nim
license: MIT
compatibility: opencode
metadata:
  audience: systems-developers
  workflow: file-io
---

## What I do

The `memfiles` module provides memory-mapped file access:
- Map files directly into memory address space
- Access file contents via pointers without copying
- Zero-copy reading and writing
- Fast iteration over lines or delimited records
- Cross-platform (Windows and POSIX)

## When to use me

Use memfiles when you need to:
- Read large files efficiently (safetensors, models, datasets)
- Access file data without copying into memory buffers
- Random access to specific file offsets
- Process lines/records in large files with zero-copy
- Share memory between processes

## Core types

### MemFile

```nim
type MemFile* = object
  mem*: pointer      ## Pointer to mapped memory
  size*: int         ## Size of mapped region

  ## Platform-specific fields:
  when defined(windows):
    fHandle*, mapHandle*: Handle
    wasOpened*: bool
  else:
    handle*: cint
    flags: cint
```

### MemSlice

```nim
type MemSlice* = object
  data*: pointer      ## Pointer to data
  size*: int          ## Size in bytes
```

MemSlice provides a view into memory-mapped data without ownership:
- Used for zero-copy access to tensor data in safetensors
- Can be cast to typed pointers for element access
- Lifetime tied to the parent MemFile

```nim
# Creating MemSlice from MemFile
var mf = memfiles.open("tensor.safetensors")
defer: mf.close()

let dataOffset = 8 + headerSize.nextMultipleOf(8)
let slice = MemSlice(data: mf.mem + dataOffset, size: tensorSize)

# Access as typed array
let floatData = cast[ptr UncheckedArray[float32]](slice.data)

# Safe access with bounds checking
if index < slice.size div sizeof(float32):
  echo floatData[index]
```

## Opening files

```nim
var mf = memfiles.open("/path/to/file.bin")

# With write access
var mf = memfiles.open("/path/to/file.bin", mode = fmReadWrite)

# Create new file with specific size
var mf = memfiles.open("/path/to/new.bin", mode = fmWrite, newFileSize = 1024)

# Map only a portion
var mf = memfiles.open("/path/to/file.bin", mappedSize = 512, offset = 0)

mf.close()
```

## Direct pointer access (zero-copy)

```nim
var mf = memfiles.open("model.safetensors")
defer: mf.close()

# Access memory directly
let ptr = cast[ptr byte](mf.mem)
let firstByte = ptr[0]

# Access as different types
let ptr32 = cast[ptr uint32](mf.mem)
let value = ptr32[0]

# Slice notation via MemSlice
let slice = MemSlice(data: mf.mem, size: mf.size)
```

## Reading binary data

```nim
var mf = memfiles.open("data.bin")
defer: mf.close()

# Read header (first 8 bytes as uint64)
let headerSize = cast[ptr uint64](mf.mem)[0]

# Read from offset
let offset = 8
let dataPtr = cast[ptr byte](mf.mem)[offset]
let value = cast[ptr uint32](cast[int](mf.mem) + offset)[0]

# Create views
type Header = object
  magic*: uint32
  version*: uint16
  flags*: uint16

let header = cast[ptr Header](mf.mem)[0]
```

## Working with safetensors

```nim
import std/tables

var mf = memfiles.open("model.safetensors")
defer: mf.close()

# Read 8-byte header size
let headerSize = cast[ptr uint64](mf.mem)[0]

# Parse JSON header (zero-copy, just cast string)
let jsonOffset = 8
let jsonPtr = cast[cstring](mf.mem) + jsonOffset
let jsonHeader = $JsonNode.fromJson($jsonPtr)

# Tensor data starts after header + padding
let dataOffset = 8 + headerSize.nextMultipleOf(8)
let tensorData = cast[ptr byte](mf.mem) + dataOffset

# Access tensor at specific offset
let tensorOffset = tensorData + tensorInfo.dataOffsets.start
let tensorPtr = cast[ptr float32](tensorOffset)
```

## Iteration with memSlices

```nim
var mf = memfiles.open("large_file.txt")
defer: mf.close()

# Iterate over lines (handles Unix \n and Windows \r\n)
for slice in memSlices(mf):
  if slice.size > 0:
    let line = cast[cstring](slice.data)
    echo line[0..<slice.size]

# Custom delimiter
for slice in memSlices(mf, delim = '\0'):
  processRecord(slice.data, slice.size)
```

## Stream interface

```nim
import std/streams

let stream = newMemMapFileStream("file.bin")
defer: stream.close()

let data = stream.readStr(1024)
stream.setPosition(0)
```

## Writing

```nim
var mf = memfiles.open("output.bin", mode = fmWrite, newFileSize = 1024)
defer: mf.close()

# Write via pointer
let ptr = cast[ptr uint32](mf.mem)
ptr[0] = 0xDEADBEEF.uint32

# Flush to disk
mf.flush()
```

## Resizing (remapping)

```nim
var mf = memfiles.open("large_file.bin", mode = fmReadWrite)
defer: mf.close()

# Resize and remap
mf.resize(2048)  # Pointer may change!

# Pointer is now invalid, need to re-get
let newPtr = cast[ptr byte](mf.mem)
```

## File size handling

```nim
var mf = memfiles.open("file.bin")

# Size is available after open
echo mf.size

# Can check file size before opening
let size = getFileSize("file.bin")
```

## Error handling

```nim
try:
  var mf = memfiles.open("nonexistent.bin")
  defer: mf.close()
except OSError as e:
  echo "Failed to open: ", e.msg
```

## Performance patterns

### Zero-copy tensor loading

```nim
proc loadTensorView*(mf: MemFile, offset, size: int): ptr UncheckedArray[byte] =
  cast[ptr UncheckedArray[byte]](cast[int](mf.mem) + offset)

var mf = memfiles.open("safetensors.bin")
defer: mf.close()

let data = mf.loadTensorView(tensorInfo.dataOffsets.start, tensorSize)
# No copy - pointer directly into memory-mapped file
```

### Large file line processing

```nim
var count = 0
for slice in memSlices(mf):
  if slice.size > 0 and cast[cstring](slice.data)[0] != '#':
    inc(count)
```

### Memory-efficient streaming

```nim
# Process in chunks without full file in memory
const CHUNK_SIZE = 1024 * 1024  # 1MB
var offset = 0
while offset < mf.size:
  let chunkSize = min(CHUNK_SIZE, mf.size - offset)
  let chunkPtr = cast[ptr byte](mf.mem) + offset
  processChunk(chunkPtr, chunkSize)
  offset += chunkSize
```

## Platform notes

- **Windows**: Uses `CreateFileMapping` and `MapViewOfFileEx`
- **POSIX**: Uses `mmap` with `MAP_SHARED`
- **Page size**: `offset` must be multiple of OS page size (usually 4K or 8K)
- **Flush**: Call `flush()` to write changes back to disk

## Common patterns with safetensors

```nim
# Load safetensors header (8 bytes size + JSON + padding)
var mf = memfiles.open("model.safetensors")
defer: mf.close()

# Header size is always little-endian uint64
let headerSize = cast[ptr uint64](mf.mem)[0]

# JSON starts at offset 8
let jsonPtr = cast[cstring](mf.mem) + 8

# Parse with jsony (creates copy, but header is small)
let header = jsonPtr[0..<headerSize.int].fromJson(JsonHeader)

# Tensor data starts at 8 + padded header size
let dataOffset = 8 + headerSize.nextMultipleOf(8)
let tensorPtr = cast[ptr byte](mf.mem) + dataOffset

# Load specific tensor (zero-copy view)
let tInfo = header.tensors["weight"]
let tensorView = cast[ptr float32](tensorPtr + tInfo.dataOffsets.start)
```

## Limitations

- Empty files cannot be memory-mapped (check size first)
- Append mode not supported (`fmAppend` raises error)
- `offset` must be multiple of OS page size
- On 32-bit systems, very large files may not be fully mappable
