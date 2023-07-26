libtargets = []

if cpp.has_argument('-march=skylake-avx512')
  libtargets += static_library('libskx',
    files(
      'x86simdsort-skx.cpp',
      ),
    include_directories : [src],
    cpp_args : ['-O3', '-mavx512f', '-mavx512dq', '-mavx512vl'],
    )
endif

if cpp.has_argument('-march=icelake-client')
  libtargets += static_library('libicl',
    files(
      'x86simdsort-icl.cpp',
      ),
    include_directories : [src],
    cpp_args : ['-O3', '-mavx512f', '-mavx512vbmi2', '-mavx512bw', '-mavx512vl', '-mf16c'],
    )
endif

if cancompilefp16
  libtargets += static_library('libspr',
    files(
      'x86simdsort-spr.cpp',
      ),
    include_directories : [src],
    cpp_args : ['-O3', '-mavx512f', '-mavx512fp16', '-mavx512vbmi2'],
    )
endif
