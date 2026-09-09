# Vendored zstd

The zstd C library is vendored as a git submodule at
`workspace/zstd/vendor/zstd`, pinned to upstream facebook/zstd. The
parent repository records the pinned commit; no zstd source file is
copied into the parent tree. `.gitmodules` at the repository root
names the URL, `git submodule status` prints the pin.

- URL: `https://github.com/facebook/zstd`
- Pinned: `f8745da6ff1ad1e7bab384bd1f9d742439278e99`, release tag v1.5.7
- Version note: the pinned `lib/zstd.h` carries
  `ZSTD_VERSION_MAJOR 1 / MINOR 5 / RELEASE 7`,
  every compiled consumer locks that version through the headers it
  reads from the submodule
- License: BSD + GPLv2 dual, `COPYING` inside the submodule

## Compile set

Only the `lib/` subtree is compiled, nothing else in the submodule
participates in a build:

- `lib/zstd.h`, `lib/zstd_errors.h`, `lib/zdict.h` — public headers
- `lib/common/`, `lib/compress/`, `lib/decompress/`, `lib/dictBuilder/`
  — the 30 `.c` files enumerated in `workspace/zstd/zstd.nim`
- `lib/legacy/` and `lib/deprecated/` stay out, matching upstream's own
  single-file build flow, which skips them too
- `lib/decompress/huf_decompress_amd64.S` ships with the tree and is
  not compiled, it is x86-64 BMI2 assembly and off by default

Internal includes are relative (`../common/zstd_internal.h`),
so the submodule checkout must stay pristine, no local patches.

## Version bump ritual

- fetch the new tag into the submodule:
  `git -C workspace/zstd/vendor/zstd fetch --depth 1 origin tag <tag>`
- move the pointer:
  `git -C workspace/zstd/vendor/zstd checkout <tag>`
- stage the new sha in the parent repo:
  `git add workspace/zstd/vendor/zstd`
- update the pinned line above and derive the version note from the
  new `lib/zstd.h` (`ZSTD_VERSION_MAJOR/MINOR/RELEASE`)
- re-run the round-trip test, `workspace/zstd/tests/t_zstd_roundtrip.nim`

## Compiled as C++

Under nim cpp the transformer suites compile every vendored source with
the C++ driver (`-std=c++20` reaches every compiled file).
config.nims tells each source its true language with `-x c++`.
At v1.5.7 the 30 `.c` files of the compile set compile clean as C++
(verified, see the round-trip test), and they carry zero basename
collisions so per-file `{.compile}` object names cannot collide in
nimcache. Should a future bump break either property, the tuple form
`{.compile: (glob, "prefix_$#.o")}` gives each object a unique name.
