## CPU feature detection via workspace/cpuplatforms/x86/cpudetect_x86.
## The module auto-runs detectCpuFeaturesX86() at load time ({.loadTime.}).

import workspace/cpuplatforms/x86/cpudetect_x86

type CPUFeatureX86* = enum
  x86_Generic
  x86_SSE
  x86_SSE2
  x86_SSE4_1
  x86_AVX
  x86_AVX_FMA
  x86_AVX2
  x86_AVX512

proc x86_cpu_name*(): string = cpuName_x86()

# Direct wrappers to Constantine's has*() procs.
# Called inside {.cast(noSideEffect).} blocks where used.

func cpuinfo_has_x86_avx512f*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasAvx512f()

func cpuinfo_has_x86_avx512bw*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasAvx512bw()

func cpuinfo_has_x86_avx512dq*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasAvx512dq()

func cpuinfo_has_x86_avx512cd*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasAvx512cd()

func cpuinfo_has_x86_avx512vbmi*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasAvx512vbmi()

func cpuinfo_has_x86_avx512vbmi2*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasAvx512vbmi2()

func cpuinfo_has_x86_avx512vl*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasAvx512vl()

func cpuinfo_has_x86_fma3*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasFma3()

func cpuinfo_has_x86_avx*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasAvx()

func cpuinfo_has_x86_avx2*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasAvx2()

func cpuinfo_has_x86_sse*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasSse()

func cpuinfo_has_x86_sse2*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasSse2()

func cpuinfo_has_x86_sse41*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasSse41()

func cpuinfo_has_x86_sse42*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasSse42()

func cpuinfo_has_x86_aes_ni*(): bool {.inline.} =
  {.cast(noSideEffect).}:
    hasAes()

proc simd_supported_hw*(cpu: CPUFeatureX86): bool {.inline.} =
  case cpu:
    of x86_Generic: true
    of x86_SSE:     hasSse()
    of x86_SSE2:    hasSse2()
    of x86_SSE4_1:  hasSse41()
    of x86_AVX:     hasAvx()
    of x86_AVX_FMA: hasAvx() and hasFma3()
    of x86_AVX2:    hasAvx2()
    of x86_AVX512:  hasAvx512f()
