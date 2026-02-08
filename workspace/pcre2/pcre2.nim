# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# This file statically compiles PCRE2 and exposes its API in Nim
# We don't modify or path the original library
# so that version updates only need:
# - git checkout
# - copying vendor/pcre2/src/pcre2.h.in to ./pcre2.h·
#   and editing versioning macro
#     ```
#     #define PCRE2_MAJOR           @PCRE2_MAJOR@
#     #define PCRE2_MINOR           @PCRE2_MINOR@
#     #define PCRE2_PRERELEASE      @PCRE2_PRERELEASE@
#     #define PCRE2_DATE            @PCRE2_DATE@
#     ```
# - potentially passing new BUILD ENV variables
#
# REFERENCES
#   - workspace/pcre2/vendor/pcre2/src/config-cmake.h.in
#   - workspace/pcre2/vendor/pcre2/build.zig
#   - workspace/pcre2/vendor/pcre2/NON-AUTOTOOLS-BUILD
#
# BUILD ENV are passed through config.nims
# to restrict them on a per-file basis,
# {.localpassc: "-D...".} would only work for the current nim file
#
# We also copy workspace/pcre2/vendor/pcre2/src/pcre2_chartables.c.dist
# to workspace/pcre2/vendor/pcre2_chartables.c
# Symlinking instead might break on windows or need extra permissions
# and `git clone -c core.symlinks=true https://github.com/mratsim/tattletale`
# to enable cloning with symlink

{.compile:"vendor/pcre2_chartables.c".}
{.compile:"vendor/pcre2/src/pcre2_auto_possess.c".}
{.compile:"vendor/pcre2/src/pcre2_chkdint.c".}
{.compile:"vendor/pcre2/src/pcre2_compile.c".}
{.compile:"vendor/pcre2/src/pcre2_compile_cgroup.c".}
{.compile:"vendor/pcre2/src/pcre2_compile_class.c".}
{.compile:"vendor/pcre2/src/pcre2_config.c".}
{.compile:"vendor/pcre2/src/pcre2_context.c".}
{.compile:"vendor/pcre2/src/pcre2_convert.c".}
{.compile:"vendor/pcre2/src/pcre2_dfa_match.c".}
{.compile:"vendor/pcre2/src/pcre2_error.c".}
{.compile:"vendor/pcre2/src/pcre2_extuni.c".}
{.compile:"vendor/pcre2/src/pcre2_find_bracket.c".}
{.compile:"vendor/pcre2/src/pcre2_jit_compile.c".}
{.compile:"vendor/pcre2/src/pcre2_maketables.c".}
{.compile:"vendor/pcre2/src/pcre2_match.c".}
{.compile:"vendor/pcre2/src/pcre2_match_data.c".}
{.compile:"vendor/pcre2/src/pcre2_match_next.c".}
{.compile:"vendor/pcre2/src/pcre2_newline.c".}
{.compile:"vendor/pcre2/src/pcre2_ord2utf.c".}
{.compile:"vendor/pcre2/src/pcre2_pattern_info.c".}
{.compile:"vendor/pcre2/src/pcre2_script_run.c".}
{.compile:"vendor/pcre2/src/pcre2_serialize.c".}
{.compile:"vendor/pcre2/src/pcre2_string_utils.c".}
{.compile:"vendor/pcre2/src/pcre2_study.c".}
{.compile:"vendor/pcre2/src/pcre2_substitute.c".}
{.compile:"vendor/pcre2/src/pcre2_substring.c".}
{.compile:"vendor/pcre2/src/pcre2_tables.c".}
{.compile:"vendor/pcre2/src/pcre2_ucd.c".}
{.compile:"vendor/pcre2/src/pcre2_valid_utf.c".}
{.compile:"vendor/pcre2/src/pcre2_xclass.c".}
