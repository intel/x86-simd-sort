libtests = []

libtests += static_library('tests_qsort',
  files('test-qsort.cpp', ),
  dependencies: gtest_dep,
  include_directories : [lib, utils],
  )

libtests += static_library('tests_kvsort',
  files('test-keyvalue.cpp', ),
  dependencies: gtest_dep,
  include_directories : [lib, utils],
  )

#if cancompilefp16
#  libtests += static_library('tests_qsortfp16',
#    files('test-qsortfp16.cpp', ),
#    dependencies: gtest_dep,
#    include_directories : [src, utils],
#    cpp_args : ['-O3', '-march=sapphirerapids'],
#    )
#endif
