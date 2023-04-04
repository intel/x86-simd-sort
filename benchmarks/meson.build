libbench = []

if cpp.has_argument('-march=icelake-client')
  libbench += static_library('bench_qsort',
    files('bench_qsort.cpp', ),
    dependencies: gbench_dep,
    include_directories : [src, utils],
    cpp_args : ['-O3', '-march=icelake-client'],
    )
endif

if cancompilefp16
  libbench += static_library('bench_qsortfp16',
    files('bench_qsortfp16.cpp', ),
    dependencies: gbench_dep,
    include_directories : [src, utils],
    cpp_args : ['-O3', '-march=sapphirerapids'],
    )
endif
