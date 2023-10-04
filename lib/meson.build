libtargets = []

if cpp.has_argument('-march=skylake-avx512')
  libtargets += static_library('libskx',
    files(
      'x86simdsort-skx.cpp',
      ),
    include_directories : [src],
    cpp_args : ['-march=skylake-avx512', flags_hide_symbols],
    )
endif

if cpp.has_argument('-march=icelake-client')
  libtargets += static_library('libicl',
    files(
      'x86simdsort-icl.cpp',
      ),
    include_directories : [src],
    cpp_args : ['-march=icelake-client', flags_hide_symbols],
    )
endif

if cancompilefp16
  libtargets += static_library('libspr',
    files(
      'x86simdsort-spr.cpp',
      ),
    include_directories : [src],
    cpp_args : ['-march=sapphirerapids', flags_hide_symbols],
    )
endif
