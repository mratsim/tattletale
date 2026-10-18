# Toktoktok test suites

- writing rules live in [testing skill](../../../.agents/skills/testing/SKILL.md)

## Test file header

Every test file opens with the same shape, nothing else:

```
## <title, one line>
##
## Test that <the single behavior the test asserts>
##
## Run:
##   <run command, verbatim, copy-paste runnable>
```

- the title names the surface under test, one line
- `Test that` states the one important behavior the test asserts,
  not the line shape
- the run command is the exact command that runs the test
- the linter enforces the shape (`test-header-testthat`)
