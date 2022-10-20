libtests = []

if cc.has_argument('-march=icelake-client')
  libtests += static_library(
    'tests_',
    files(
      'test_all.cpp',
      ),
    dependencies : gtest_dep,
    include_directories : [
      src,
      utils,
      ],
    cpp_args : [
      '-O3',
      '-march=icelake-client',
      ],
    )
endif
