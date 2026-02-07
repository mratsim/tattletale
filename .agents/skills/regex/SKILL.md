---
name: regex
description: Regex functionality in Nim including std/re, std/nre wrappers around PCRE, and the pure Nim nim-regex alternative with linear-time matching guarantees
---

# Regex Skill for Nim

This skill covers regex functionality in Nim, including the standard library modules (`std/re`, `std/nre`), the pure Nim alternative (nim-regex), and the context around PCRE vs PCRE2 migration.

## Overview of Nim's Regex Ecosystem

Nim provides multiple regex implementations:

1. **std/re**: Legacy wrapper around PCRE (Perl-Compatible Regular Expressions)
2. **std/nre**: Modern wrapper around PCRE with better API design
3. **nim-regex**: Pure Nim implementation (drop-in replacement, linear-time matching)

### The PCRE vs PCRE2 Context

Nim's standard library currently depends on PCRE (not PCRE2), which:
- Last release was in 2021 (no longer actively maintained)
- Is being deprecated in Debian stable (see [Debian bug #1071970](https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1071970))
- Will be phased out in future distributions

The issue [#23668](https://github.com/nim-lang/Nim/issues/23668) tracks the migration from PCRE to PCRE2. However, this is a significant undertaking since PCRE and PCRE2 are API-incompatible.

For users wanting to avoid PCRE dependency, **nim-regex** provides a pure Nim alternative.

---

## std/re - Legacy PCRE Wrapper

The original regex module wrapping the PCRE C library.

### Creating Regex Patterns

```nim
import std/re

# Basic pattern
let emailPattern = re"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

# With flags
let caseInsensitive = re"pattern", {reIgnoreCase}
let multiLine = re"^pattern", {reMultiLine}
let dotAll = re"pattern.*", {reDotAll}
let extended = re"pattern", {reExtended}  # Ignores whitespace and comments
let studied = re"pattern", {reStudy}  # Pre-analyzes for performance

# Extended regex with comments
let verbose = rex"""(?x)
  \d+    # Match digits
  .*     # Followed by anything
"""
```

### Matching Operations

```nim
import std/re

# Simple match check
if match("hello world", re"hello"):
  echo "Found hello"

# Match with captures
var matches: array[3, string]
if match("abc123def", re"([a-z]+)(\d+)([a-z]+)", matches):
  echo matches[0]  # "abc"
  echo matches[1]  # "123"
  echo matches[2]  # "def"

# Find first occurrence
let pos = find("abcdefg", re"cde")  # Returns 2
let posNotFound = find("abcdefg", re"xyz")  # Returns -1

# Find with captures
var captures: array[2, string]
let foundPos = find("key=value; key2=value2", re"(\w+)=(\w+)", captures)
# captures[0] = "key", captures[1] = "value"

# Match length
let len = matchLen("abcdefg", re"cde", 2)  # Returns 3
let lenNotFound = matchLen("abcdefg", re"xyz")  # Returns -1

# Check if string contains pattern
if contains("abcdef", re"cde"):
  echo "Pattern found"

# Starts with / Ends with
if startsWith("hello world", re"hello"):
  echo "Starts with hello"
if endsWith("hello world", re"world"):
  echo "Ends with world"
```

### Find All Iterations

```nim
import std/re

# Iterator version
for word in findAll("the quick brown fox", re"\w+"):
  echo word  # "the", "quick", "brown", "fox"

# Seq version
let allMatches = findAll("abcabcabc", re"abc")  # @["abc", "abc", "abc"]
```

### Replace Operations

```nim
import std/re

# Simple replace
let replaced = "foo bar foo".replace(re"foo", "baz")
# "baz bar baz"

# Replace with captures
let replacedWithCaptures = "var1=key; var2=key2".replacef(re"(\w+)=(\w+)", "$1<-$2$2")
# "var1<-keykey; var2<-key2key2"

# Multi-replace in parallel
let multiResult = "abc123xyz".multiReplace([
  (re"\d+", "NUM"),
  (re"[a-z]+", "LET")
])
```

### String Splitting

```nim
import std/re, std/sequtils

let parts = toSeq(split("a1b2c3", re"\d+"))
# @["a", "b", "c"]

let withCaptures = toSeq(split("a1b2", re"(\d)"))
# @["a", "1", "", "b", "2", ""]
```

### =~ Macro (Implicit Matches)

```nim
import std/re

if "NAME = VALUE" =~ re"\s*(\w+)\s*=\s*(\w+)":
  echo matches[0], " = ", matches[1]  # NAME = VALUE

elif "   # comment" =~ re"\s*(\#.*)":
  echo matches[0]  # "# comment"
```

---

## std/nre - Modern PCRE Wrapper

Improved API over PCRE with Option-based returns and better ergonomics.

### Creating Regex Patterns

```nim
import std/nre

# Basic pattern
let pattern = re"(\w+)@(\w+)\.(\w+)"

# With inline flags
let unicodePattern = re"(?i)hello"  # Case insensitive
let multilinePattern = re"(?m)^start"  # ^ matches line beginnings
let dotallPattern = re"(?s)pattern.*"  # . matches newlines
let extendedPattern = re"(?x)pattern # comment"  # Whitespace ignored
let ungreedyPattern = re"(?U)pattern.*"  # Lazy quantification

# Pattern options (at start)
let utf8Pattern = re"(*UTF8)pattern"  # Treat as UTF-8
let ucpPattern = re"(*UCP)\w+"  # Unicode character properties
let crlfNewlines = re"(*CRLF)pattern"  # CRLF line endings
let noAutoCapture = re"(*NO_AUTO_CAPTURE)(?<name>\w+)"  # Manual captures only
```

### Matching Operations

```nim
import std/nre, std/options

# Try to match entire string
let result = match("foobar", re"foobar")
# Result: Option[RegexMatch] = some(RegexMatch(...))

if result.isSome:
  let m = result.get()
  echo m.match  # "foobar"
  echo m.captures[0]  # captured group if any

# Match with captures
let captureResult = match("john@example.com", re"(\w+)@(\w+)\.(\w+)")
if captureResult.isSome:
  let m = captureResult.get()
  echo m.captures[0]  # "john"
  echo m.captures[1]  # "example"
  echo m.captures[2]  # "com"

# Named captures
let namedResult = match("john@example.com", re"(?<user>\w+)@(?<domain>\w+)")
if namedResult.isSome:
  let m = namedResult.get()
  echo m.captures["user"]    # "john"
  echo m.captures["domain"]  # "example"

# Find substring
let findResult = find("email: john@example.com", re"(\w+)@(\w+)")
if findResult.isSome:
  let m = findResult.get()
  echo m.match  # "john@example.com"
  echo m.captures[0]  # "john"
```

### Iteration with findAll

```nim
import std/nre

# Find all matches
for m in findIter("one two three four", re"\w+"):
  echo m.match  # "one", "two", "three", "four"

# With captures
for m in findIter("a1b2c3", re"(\w)(\d)"):
  echo m.captures[0], "-", m.captures[1]  # "a-1", "b-2", "c-3"

# Get all as seq
let all = findAll("abcabc", re"abc")
# @["abc", "abc"]
```

### Accessing Captures and Bounds

```nim
import std/nre

let m = find("test@example.com", re"(\w+)@(\w+)").get()

# Captures by index
echo m.captures[0]     # "test"
echo m.captures[1]     # "example"
echo m.captures[-1]    # Full match: "test@example.com"

# Capture bounds (inclusive range)
echo m.captureBounds[0]  # 0 .. 3
echo m.captureBounds[1]  # 5 .. 11

# Check if capture group was matched
if 0 in m.captureBounds:
  echo "Group 0 matched"

# Convert to table
let namedResult = match("key=value", re"(?<k>\w+)=(?<v>\w+)").get()
let table = namedResult.captures.toTable()
# {"k": "key", "v": "value"}

# Convert to seq
let seqResult = namedResult.captures.toSeq()
# @["key", "value"]
```

### Splitting Strings

```nim
import std/nre

# Basic split
let parts = split("a1b2c3", re"\d+")
# @["a", "b", "c"]

# With captures included
let partsWithCaptures = split("a1b2", re"(\d)")
# @["a", "1", "", "b", "2", ""]

# Max splits
let limited = split("a1b2c3d4", re"\d", maxSplit = 2)
# @["a", "b", "c3d4"]
```

### Replace Operations

```nim
import std/nre

# With proc replacement
let upperResult = replace("hello world", re"\w+", proc(m: RegexMatch): string =
  m.match.toUpperAscii()
)
# "HELLO WORLD"

# With string replacement (captures with $N notation)
let formatted = replace("john@email.com", re"(\w+)@(\w+)", "$1 <at> $2")
# "john <at> email"

# Named captures
let namedReplaced = replace("key=value", re"(?<k>\w+)=(?<v>\w+)", "$k = $v")
# "key = value"

# Dollar sign escape
let dollarResult = replace("price: $100", re"\$\d+", "$$$")
# "price: $$"
```

### Contains Check

```nim
import std/nre

if contains("abcdef", re"cde"):
  echo "Contains pattern"

# With bounds
if contains("abcdef", re"cde", start = 1):
  echo "Contains from position 1"
```

---

## nim-regex - Pure Nim Drop-in Replacement

Pure Nim regex implementation with linear-time matching guarantees. Designed as a drop-in replacement for std/re and std/nre.

### Why nim-regex?

1. **No C dependencies**: Pure Nim, compiles to JavaScript/WebAssembly easily
2. **Linear-time matching**: O(n) complexity, safe for untrusted input
3. **No backreferences**: Simpler, faster, safer
4. **Drop-in replacement**: Works where PCRE is unavailable

### Creating Patterns with nim-regex

```nim
import regex

# Basic pattern (compile-time)
let pattern = re2"(\w+)@(\w+)\.(\w+)"

# Runtime compilation
let runtimePattern = re2(someString)

# With flags
let flags = {regexDotAll, regexCaseless}
let flaggedPattern = re2("pattern", flags)

# Arbitrary bytes mode (treat as bytes, not UTF-8)
let bytesPattern = re2(r"\xff\xfe", {regexArbitraryBytes})

# Raw string literal
let rawPattern = rex"""(?x)
  \d+  # digits
  .*  # anything
"""
```

### nim-regex API

```nim
import regex

# Match (whole string must match)
var m = RegexMatch2()
if match("abc", re2"abc", m):
  echo m.group(0)  # Full match bounds

# Find (substring match)
var findM = RegexMatch2()
if "abcd".find(re2"bc", findM):
  echo findM.boundaries  # 1 .. 2

# Find all
for match in findAll("abcabc", re2"abc"):
  echo match.match

# Find all bounds
for bounds in findAllBounds("abcabc", re2"bc"):
  echo bounds  # 1 .. 2, 4 .. 5

# Contains
if re2"bc" in "abcd":
  echo "Contains"

# Split
let parts = split("a1b2c3", re2"\d+")
# @["a", "b", "c"]

# Split including captures
let withCaps = splitIncl("a,b", re2"(,)")
# @["a", ",", "b"]

# Replace
let replaced = "aaa".replace(re2"a", "b", 1)  # Limit to 1 replacement
# "baa"

# With capture references
let withCaptures = "abc".replace(re2"(a)(b)c", "m($1) m($2)")
# "m(a) m(b)"

# Replace with proc
proc removeStars(m: RegexMatch2, s: string): string =
  if s[m.group(0)] == "*": ""
  else: s[m.group(0)]

let cleaned = "**test**".replace(re2"(\*)", removeStars)
# "test"

# Starts with / Ends with
if "abc".startsWith(re2"\w"):
  echo "Starts with word"
if "abc".endsWith(re2"\w"):
  echo "Ends with word"
```

### Accessing Results

```nim
import regex

var m = RegexMatch2()
discard "hello world".find(re2"(\w+) (\w+)", m)

# Groups by index
echo m.group(0)    # 0 .. 4 (full match bounds)
echo m.group(1)    # 0 .. 4 (first capture)
echo m.group(2)    # 6 .. 10 (second capture)

# Groups by name
echo m.group("word1")  # Bounds for named group

# Groups count
echo m.groupsCount  # Number of capture groups

# Group names
echo m.groupNames   # @["word1", "word2"] if named

# Captured text
let capturedText = "test string"[m.group(1)]
```

### Match Macro

```nim
import regex

# Compile-time regex with macro
match "abc", rex"(\w)+":
  echo matches  # @["c"] - last capture in repeated group

match "[link](https://example.com)", rex"\[([^\]]+)\]\((https?://[^)]+)\)":
  echo matches[0]  # "link"
  echo matches[1]  # "https://example.com"
```

### Compile-time vs Runtime Compilation

```nim
import regex

# Compile-time (static string)
const staticPattern = re2"\d+"

# Runtime (dynamic string)
let runtimePattern = re2(someString)

# Function with static parameter
func myMatch(s: string, exp: static string): bool =
  const compiled = re2(exp)
  s.match(compiled)

myMatch("123", r"\d+")  # Compiles regex at compile time
```

### Escape and Special Characters

```nim
import regex

# Escape regex special chars
let escaped = escapeRe("file.txt")
# Matches literal "file.txt", not regex pattern

# Special chars that need escaping:
# ' ', '#', '$', '&', '(', ')', '*', '+', '-', '.', 
# '?', '[', '\\', ']', '^', '{', '|', '}', '~'
```

### Unicode Considerations

```nim
import regex

# By default, Unicode friendly
assert match("弢弢弢", re2"\w+")  # Works with CJK

# ASCII mode only
assert not match("弢弢弢", re2(r"\w+", {regexAscii}))

# Invalid UTF-8 handling (debug mode validates)
when not defined(release):
  import unicode
  assert validateUtf8("valid string") == -1
  assert validateUtf8("\xf8\xa1\xa1\xa1\xa1") != -1

# Arbitrary bytes mode
let bytesFlags = {regexArbitraryBytes}
assert match("\xff\xfe", re2(r"\xff\xfe", bytesFlags))
```

---

## Quick Comparison Reference

| Feature | std/re | std/nre | nim-regex |
|---------|--------|---------|-----------|
| C Dependency | PCRE | PCRE | None |
| API Style | Return codes | Option[T] | Option[T] |
| Complexity | Varies | Varies | O(n) linear |
| Backreferences | Yes | Yes | No |
| Compile-time | No | No | Yes |
| JS Compatible | No | No | Yes |
| Drop-in Replace | Partial | Partial | Yes |

### Choosing the Right Module

- **std/re**: Legacy code, simple use cases, when PCRE is already available
- **std/nre**: Modern PCRE wrapper, better API, when you need PCRE features
- **nim-regex**: No C deps needed, linear-time guarantees, WebAssembly targets

### Common Patterns

```nim
# Email validation
const emailRe = re2"""(?x)
  [a-zA-Z0-9._%+-]+
  @
  [a-zA-Z0-9.-]+
  \.
  [a-zA-Z]{2,}
"""

# IPv4 address
const ipv4Re = re2"""(?x)
  \b
  ((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}
  (25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)
  \b
"""

# URL pattern
const urlRe = re2"""(?x)
  https?://
  [^\s]+
"""
```

---

## Error Handling

```nim
# std/nre / nim-regex errors
try:
  let badPattern = re2"[unclosed"
except RegexError:
  let e = getCurrentException()
  echo "Regex error: ", e.msg

# Specific error types (nim-regex)
try:
  discard re2(pattern)
except SyntaxError:
  echo "Invalid regex syntax at pos ", e.pos
except StudyError:
  echo "Regex study failed"
```

---

## Performance Tips

1. **Use `reStudy` flag** in std/re for repeated matches
2. **Compile patterns once**, store in variables/constants
3. **nim-regex**: Use compile-time strings (`const` or `static`)
4. **Prefer `contains`** over `find` for boolean checks
5. **Use `findAllBounds`** when you only need positions, not captures
