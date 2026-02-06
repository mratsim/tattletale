---
name: tables
description: Nim hash tables (dictionaries) with value and ref semantics
license: MIT
compatibility: opencode
metadata:
  audience: general-developers
  workflow: general-programming
---

## What I do

Nim's `tables` module provides hash table implementations:
- `Table[K, V]` - generic hash table with value semantics
- `OrderedTable[K, V]` - hash table preserving insertion order
- `CountTable[K]` - maps keys to occurrence counts
- `Ref` variants share reference semantics

## When to use me

Use tables when you need:
- Key-value mappings (dictionary behavior)
- Fast O(1) lookup by key
- Ordered insertions (use OrderedTable)
- Counting frequencies (use CountTable)

## Core types

```nim
type
  Table*[K, V] = object
    data: KeyValuePairSeq[K, V]
    counter: int

  TableRef*[K, V] = ref Table[K, V]  # Ref shares memory

  OrderedTable*[K, V] = object
    data: OrderedKeyValuePairSeq[K, V]
    counter, first, last: int

  OrderedTableRef*[K, V] = ref OrderedTable[K, V]

  CountTable*[K] = object
    data: seq[tuple[key: K, val: int]]
    counter: int
    isSorted: bool

  CountTableRef*[K] = ref CountTable[K]
```

## Creating tables

```nim
# Empty tables
let empty = initTable[string, int]()
let emptyRef = newTable[string, int]()

# From pairs
let t1 = {"a": 1, "b": 2}.toTable
let t2 = @[("x", 10), ("y", 20)].toTable

# Ordered
let ordered = [('z', 1), ('y', 2), ('x', 3)].toOrderedTable

# CountTable from container
let freq = toCountTable("abracadabra")
```

## Insertion and assignment

```nim
var t = initTable[string, int]()

# Insert or update
t["key"] = 42

# HasKeyOrPut returns bool indicating if key existed
if t.hasKeyOrPut("key", 99):
  t["key"] = t["key"] + 1  # key existed, increment
else:
  discard  # key was just inserted
```

## Lookup

```nim
let t = {"a": 1, "b": 2}.toTable

# Direct lookup - raises KeyError if missing
let val = t["a"]

# HasKey check before access
if t.hasKey("a"):
  echo t["a"]

# getOrDefault returns default (0) or custom value
let x = t.getOrDefault("missing")       # 0
let y = t.getOrDefault("missing", -1)   # -1

# contains for `in` operator
if "a" in t:
  discard
```

## Deletion

```nim
var t = {"a": 1, "b": 2, "c": 3}.toTable

t.del("a")        # Does nothing if key absent

# pop removes and returns whether it existed
var val: int
if t.pop("b", val):
  echo "removed b with value: ", val
```

## Size

```nim
let t = {"a": 1, "b": 2}.toTable
echo len(t)  # 2
```

## Iteration

```nim
let t = {"a": 1, "b": 2, "c": 3}.toTable

# Pairs - yields (key, value)
for k, v in t.pairs:
  echo k, ": ", v

# Keys only
for k in t.keys:
  echo k

# Values only
for v in t.values:
  echo v

# Mutable iteration
var t2 = {"a": @[1], "b": @[2]}.toTable
for k, v in t2.mpairs:
  v.add(v[0] + 10)
```

## OrderedTable preserves order

```nim
let ordered = [('z', 1), ('y', 2), ('x', 3)].toOrderedTable

for k, v in ordered.pairs:
  echo k, ": ", v
# Output: z: 1, y: 2, x: 3 (insertion order preserved)
```

## CountTable for frequencies

```nim
var counts = initCountTable[char]()

# Increment
for c in "abracadabra":
  counts.inc(c)

# Or create from container
let freq = toCountTable("abracadabra")

# Access count (0 if missing)
echo freq['a']  # 5

# Most/least common
let most = freq.largest    # ('a', 5)
let least = freq.smallest  # ('c', 1)

# Sort by count (destructive)
counts.sort()  # descending by default
for k, v in counts.pairs:
  echo k, ": ", v
```

## Value vs Ref semantics

```nim
# Table (value semantics - copy on assignment)
var a = {"x": 1}.toTable
var b = a
b["x"] = 99
assert a["x"] == 1   # a unchanged

# TableRef (ref semantics - shared)
var c = {"x": 1}.newTable
var d = c
d["x"] = 99
assert c["x"] == 99   # c changed too
```

## Common patterns

```nim
# Build table from zip
from std/sequtils import zip
let names = ["John", "Paul"]
let ages = [30, 25]
var table = initTable[string, int]()
for (n, a) in zip(names, ages):
  table[n] = a

# Group by key
var byYear = initTable[int, seq[string]]()
for (year, name) in zip(years, names):
  if not byYear.hasKey(year):
    byYear[year] = @[]
  byYear[year].add(name)

# Safe get-or-put with mgetOrPut
var t = initTable[string, int]()
t.mgetOrPut("counter", 0).inc
# Returns mutable reference, can modify directly
```

## Clear and reset

```nim
var t = {"a": 1, "b": 2}.toTable
clear(t)
assert len(t) == 0
```

## Error handling

```nim
let t = {"a": 1}.toTable

try:
  echo t["missing"]
except KeyError:
  echo "key not found"
```

## Performance notes

- All lookup operations are O(1) amortized
- Insertion may trigger table enlargement (amortized O(1))
- Iteration is O(n)
- `len()` is O(1)
- CountTable.sort() is destructive - don't modify after sorting
