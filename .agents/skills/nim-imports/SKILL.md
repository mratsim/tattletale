---
name: nim-imports
description: Common import patterns and pitfalls for the Tattletale Nim project
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: nim-development
---

## What I do

I provide guidelines for importing modules and common pitfalls in the Tattletale Nim project.

## Project Directory Structure

```
/home/beta/Programming/Perso/tattletale/
├── config.nims                    # Project-level config with paths and tasks
├── workspace/
    └── mylib/                    # First-party libraries
        ├── mylib.nim            # Re-exports all public symbols
        ├── src/
        │   └── mylib.nim       # Implementation
        └── tests/
            └── test_mylib.nim  # Tests
```

## Import Patterns

### First-party workspace libraries

```nim
import workspace/mylib          # Imports from mylib.nim which re-exports
```

Note: Do NOT use `import workspace/mylib/mylib` - the `.nim` extension and file path are handled by the config.nims ` --path:"."` setting.

### Third-party packages (via nimble)

```nim
import pkg/jsony               # For packages installed via nimble
```

### Standard library modules

```nim
import std/tables              # For stdlib modules
import std/os
import std/json
import std/sequtils           # IMPORTANT: Required for mapIt
```

## Common Pitfalls

### Missing std/sequtils for mapIt

Always import `std/sequtils` when using `mapIt`:

```nim
import std/sequtils  # MISSING import causes: "undeclared identifier: mapIt"
let nums = @[1, 2, 3]
let doubled = nums.mapIt(it * 2)  # Fails without sequtils import
```

### Wrong import for workspace libraries

**WRONG:**
```nim
import toktoktok               # Fails - not in stdlib or pkg
import workspace/toktoktok/toktoktok.nim  # Wrong path
```

**CORRECT:**
```nim
import workspace/toktoktok    # Imports the re-export module
```

### Confusing pkg/ vs workspace/

- `pkg/` - Third-party packages installed via nimble (e.g., jsony, chronos)
- `workspace/` - First-party project libraries

```nim
import pkg/jsony              # Third-party JSON library
import workspace/toktoktok   # Your project's tokenizer
```

## Accessing Private Procedures

To access private (non-`*`) procedures from a module, use the `{.all.}` pragma:

```nim
import workspace/toktoktok {.all.}

# Now you can access bytePairEncode and bytePairMerge even though they were private
let ranks = initTable[seq[byte], int]()
ranks[@[byte(228), byte(189)]] = 19526
let result = bytePairEncode(piece, ranks)
```

## Accessing Private Fields

For accessing private fields of objects, use `std/importutils` with `privateAccess`:

```nim
# file_a.nim
type MyObjectA* = object # public
  privateFieldA: int     # privaterivate

# file_b.nim
type MyObjectB = object  # private
  privateFieldB: int     # private

# file_c.nim
import std/importutils
import file_a            # MyObjectA is public
import file_b {.all.}    # MyObjectB is private

block:
    privateAccess(MyObjectA)
    let obj = MyObject(privateFieldA: 42)
    echo obj.privateFieldA  # Accessible in this scope
block:
    privateAccess(MyObjectB)
    let obj = MyObject(privateFieldB: 42)
    echo obj.privateFieldB  # Accessible in this scope
```

## Re-export Pattern

Each workspace library should have a re-export module at the root:

```nim
# workspace/mylib/mylib.nim
# Re-exports all public symbols from src/ and submodules

export src/mylib
```

This allows users to `import workspace/mylib` instead of knowing the internal structure.

## PCRE2 Notes

PCRE2 is vendored in `workspace/pcre2`. The config.nims automatically sets:
- Include paths for headers
- Platform-specific defines
- PCRE2-specific configuration

No manual configuration needed - just `import workspace/pcre2`.

## Config.nims Setup

The project root has a `config.nims` that sets:
- `--path:"."` - Allows `import workspace/foo` syntax
- PCRE2 compile flags
- Test task commands

Tests are discovered by:
- Files in `workspace/*/tests/` directory
- Filenames starting with `test_` or `t_`
