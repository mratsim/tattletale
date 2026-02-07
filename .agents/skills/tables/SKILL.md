---
name: tables
description: Nim's hash table module for key-value storage
license: MIT
compatibility: opencode
metadata:
  audience: nim-developers
  use-case: data-structures
---

# Nim `tables` Module

The `tables` module provides hash table implementations for key-value storage in Nim.

## Table Types

- **`Table[A, B]`** - Standard hash table (value semantics, copies on assignment)
- **`TableRef[A, B]`** - Reference-based hash table (shared on assignment)
- **`OrderedTable[A, B]`** - Preserves insertion order
- **`CountTable[A]`** - Maps keys to occurrence counts

## Creating Tables

```nim
import std/tables

# Create empty table
var t = initTable[string, int]()
t["key"] = 42

# Create from pairs literal
let t2 = {"a": 1, "b": 2}.toTable

# TableRef for ref semantics (shared state)
let ref_t = newTable[string, int]()
ref_t["key"] = 42
```

## Basic Operations

```nim
# Insert or update
t["key"] = 42

# Access (raises KeyError if missing)
let val = t["key"]

# Check if key exists
if t.hasKey("key"):
  discard

# Get with default value
let val = t.getOrDefault("key", 0)

# Atomic check-and-set
if t.hasKeyOrPut("key", defaultValue):
  # key already existed
else:
  # key was just inserted

# Get or modify
discard t.mgetOrPut("key", defaultValue)
t["key"] = t["key"] + 1

# Delete (does nothing if missing)
t.del("key")

# Length
echo t.len
```

## Iteration

Use `pairs`, `keys`, and `values` iterators to traverse tables:

```nim
for k, v in t.pairs:
  echo "key: ", k, " value: ", v

for k in t.keys:
  echo "key: ", k

for v in t.values:
  echo "value: ", v

# For mutable tables, use mpairs/mvalues to modify in place
for k, v in t.mpairs:
  v = v + 1
```

## OrderedTable

Preserves insertion order (unlike regular Table):

```nim
import std/tables

var ot = initOrderedTable[string, int]()
ot["z"] = 1
ot["a"] = 2
ot["m"] = 3

# Iteration follows insertion order: z, a, m
for k, v in ot.pairs:
  echo k, " -> ", v
```

## CountTable

Counts occurrences (useful for frequency analysis):

```nim
import std/tables

# Create from sequence
var ct = toCountTable("abracadabra")
# Result: {'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1}

# Increment count
ct.inc('x')
ct.inc('y', 5)  # increment by 5

# Get count (returns 0 if missing)
echo ct['a']  # 5

# Sort by frequency
ct.sort()  # descending by default
```

## Common Patterns

### Safe dictionary access pattern

```nim
template withValue*[A, B](t: var Table[A, B], key: A, value, body: untyped) =
  ## Retrieves value at t[key] if it exists, binds to `value`
  mixin rawGet
  var hc: Hash
  var index = rawGet(t, key, hc)
  let hasKey = index >= 0
  if hasKey:
    var value {.inject.} = addr(t.data[index].val)
    body

# Usage
t.withValue("mykey", val):
  echo "Found: ", val
do:
  echo "Key not found"
```

### Merge two CountTables

```nim
var result = ct1
for k, v in ct2:
  result.inc(k, v)
```

## Semantics Differences

| Feature | `Table` | `TableRef` |
|---------|---------|------------|
| Assignment | Copies entire table | Shares reference |
| Memory | Each copy is independent | All refs point to same data |
| Use when | Isolation needed | Shared mutable state |

## Notes

- Tables use `hash` proc for keys - works with int, string, and custom types with defined `hash` proc
- `pairs` iterator returns `(key, value)` tuples - use `toSeq()` to convert to seq
- `OrderedTable` uses more memory but preserves insertion order
- `CountTable` uses zero as sentinel, so count of 0 means "empty slot"
