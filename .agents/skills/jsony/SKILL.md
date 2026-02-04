---
name: jsony
description: Fast JSON parsing and serialization for Nim with automatic derived hooks
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: data-parsing
---

## What I do

jsony is a high-performance JSON library for Nim that provides:
- Automatic derivation of JSON parsers/serializers for custom types
- Customizable hooks for fine-grained control (`parseHook`, `dumpHook`, `renameHook`, etc.)
- Zero-copy parsing where possible
- No external dependencies

## When to use me

Use jsony when you need to:
- Parse JSON into Nim types (objects, seqs, arrays, tables, enums, tuples)
- Serialize Nim types to JSON
- Handle nested or complex JSON structures
- Customize field names, default values, or validation

## Basic usage

### Parsing JSON into types

```nim
import pkg/jsony

type MyObject = object
  name: string
  value: int

let json_str = """{"name": "test", "value": 42}"""
let obj = json_str.fromJson(MyObject)
```

### Serializing types to JSON

```nim
let obj = MyObject(name: "test", value: 42)
let json_str = obj.toJson()
```

## Supported types

- **seq[T]** - Parses JSON arrays into sequences
- **array[N, T]** - Fixed-size arrays
- **object** - JSON objects map to object fields
- **ref object** - Pointer types
- **enum** - String or integer enum values
- **tuple** - Named or unnamed tuples
- **Table[K, V]** / **OrderedTable** - JSON objects
- **set[T]** / **HashSet** - JSON arrays
- **Option[T]** - Nullable values (null maps to `none`)
- **bool, int, float, string, char** - Primitives
- **JsonNode** - Raw JSON parsing

## Custom hooks for fine-grained control

### renameHook - Map JSON keys to different field names

```nim
proc renameHook*(v: var MyObject, key: string) =
  if key == "my_name":
    key = "name"
```

### enumHook - Custom enum parsing

```nim
proc enumHook*(strVal: string, v: var MyEnum) =
  if strVal == "ACTIVE":
    v = MyEnum.active
```

### newHook - Initialize fields with defaults

```nim
proc newHook*(v: var MyObject) =
  v.value = 100  # Default value
```

### postHook - Validate after parsing

```nim
proc postHook*(v: var MyObject) =
  if v.value < 0:
    raise newException(ValueError, "Value must be positive")
```

### skipHook - Skip specific fields during parsing

```nim
proc skipHook*(T: typedesc[MyObject], key: string): bool =
  key == "internal_field"  # Skip this field
```

## Example from Constantine tests

From `t_ethereum_bls_signatures.nim` and `t_ethereum_evm_precompiles.nim`:

```nim
import pkg/jsony

type
  PubkeyField = object
    pubkey: array[48, byte]
  DeserG1_test = object
    input: PubkeyField
    output: bool

# Custom parsing for byte arrays from hex strings
proc parseHook*[N: static int](src: string, pos: var int, value: var array[N, byte]) =
  var str: string
  parseHook(src, pos, str)
  value.paddedFromHex(str, bigEndian)

# Parse test vectors
let testFile = readFile(testDir / filename)
let testData = testFile.fromJson(DeserG1_test)

# Use parsed data
let status = pubkey.deserialize_pubkey_compressed(testData.input.pubkey)
```

## Dealing with extra JSON fields

By default, jsony ignores unknown fields:
- `fromJson[T]()` ignores extra JSON fields
- Missing fields keep their default values (use `newHook` to set defaults)

## Performance tips

- For large JSON, use `parseHook` instead of generic parsing
- `distinct` types automatically derive serialization
- Compile with `-d:release` for optimized parsing

## Error handling

```nim
try:
  let obj = json_str.fromJson(MyObject)
except JsonError as e:
  echo "JSON parse error: ", e.msg
```

## Recursive parseHook - Nested complex structures

From `t_hash_to_curve.nim` and `t_ec_sage_template.nim`:

```nim
type
  EC_G1_hex = object
    x: string
    y: string

  EC_G2_hex = object
    x: Fp2_hex  # Nested object
    y: Fp2_hex

  Fp2_hex = object
    c0: string
    c1: string

  TestVector[EC: EC_ShortW_Aff] = object
    P: EC          # Recursive generic type
    Q0, Q1: EC
    msg: string
    u: seq[string]

  HashToCurveTest[EC: EC_ShortW_Aff] = object
    L, Z, dst: string
    curve: string
    field: FieldDesc
    vectors: seq[TestVector[EC]]

# Recursive parseHook for elliptic curve points
proc parseHook*(src: string, pos: var int, value: var EC_ShortW_Aff) =
  when EC_ShortW_Aff.F is Fp:
    var P: EC_G1_hex
    parseHook(src, pos, P)           # Recursive call
    let ok = value.fromHex(P.x, P.y)
  elif EC_ShortW_Aff.F is Fp2:
    var P: EC_G2_hex
    parseHook(src, pos, P)           # Recursive call
    let ok = value.fromHex(P.x.c0, P.x.c1, P.y.c0, P.y.c1)

# Generic bigint parsing
proc parseHook*(src: string, pos: var int, value: var BigInt) =
  var str: string
  parseHook(src, pos, str)           # Parse string first
  value.fromHex(str)                 # Then convert

# Usage with generic test vectors
let vec = loadVectors(HashToCurveTest[EC_ShortW_Aff[Fp[BLS12_381], G1]], filename)

for i in 0 ..< vec.vectors.len:
  var P: EC
  P.fromAffine(vec.vectors[i].P)     # Nested parsing
```

## Working with conditional fields and generics

From `t_ec_sage_template.nim`:

```nim
type
  ScalarMulTestG2[EC: EC_ShortW_Aff, bits: static int] = object
    curve, group, modulus: string
    # Conditional fields based on generic parameter
    when EC.F is Fp:
      non_residue_twist: int
    else:
      non_residue_twist: array[2, int]
    vectors: seq[TestVector[EC, bits]]

# Load vectors with generic type
proc loadVectors(TestType: typedesc): TestType =
  const group = when TestType.EC.G == G1: "G1" else: "G2"
  const filename = "tv_" & $TestType.EC.F.Name & "_scalar_mul_" & group & "_" & $TestType.bits & "bit.json"
  let content = readFile(TestVectorsDir/filename)
  result = content.fromJson(TestType)

# Usage
let vec = loadVectors(ScalarMulTestG2[EC_ShortW_Aff[Fp2[BLS12_381], G2], 256])
```

## Advanced patterns

### Static if/when in hooks

```nim
proc parseHook*(src: string, pos: var int, value: var MyGeneric[T]) =
  when T is SomeNumber:
    # Parse numeric type
    var num: T
    parseHook(src, pos, num)
    value = MyGeneric[T](data: num)
  elif T is string:
    # Parse string type
    var str: string
    parseHook(src, pos, str)
    value = MyGeneric[T](data: str)
```

### Multiple dispatch with static parameters

```nim
proc parseHook*[EC: static typedesc](src: string, pos: var int, value: var EC) =
  # Static dispatch based on EC type
  when EC.G == G1:
    var P: EC_G1_hex
    parseHook(src, pos, P)
    discard value.fromHex(P.x, P.y)
  else: # EC.G == G2
    var P: EC_G2_hex
    parseHook(src, pos, P)
    discard value.fromHex(P.x.c0, P.x.c1, P.y.c0, P.y.c1)
```

## Integration with other code

jsony works seamlessly with:
- Standard library types (tables, sets, options)
- Custom serialization libraries (via `parseHook`/`dumpHook`)
- Test frameworks for parsing test vectors
- FFI code for parsing C/interop data structures
- Generic parameterized types and static dispatch
