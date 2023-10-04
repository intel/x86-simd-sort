libbench = []

libbench += static_library('bench_qsort',
  files(
    'bench-all.cpp',
    ),
  dependencies: gbench_dep,
  include_directories : [src, lib, utils],
  cpp_args : ['-O3'],
  )
