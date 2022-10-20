project('x86-simd-sort', 'c', 'cpp',
        version : '1.0.0',
        license : 'BSD 3-clause')
cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
src = include_directories('./src')
bench = include_directories('./benchmarks')
utils = include_directories('./utils')
tests = include_directories('./tests')
gtest_dep = dependency('gtest', fallback : ['gtest', 'gtest_dep'])
subdir('./tests')

testexe = executable('testexe', 'tests/main.cpp',
                     dependencies : gtest_dep,
                     link_whole : [
                       libtests,
                       ]
                     )

benchexe = executable('benchexe', 'benchmarks/main.cpp',
                    include_directories : [
                      src,
                      utils,
                      bench,
                      ],
                    cpp_args : [
                      '-O3',
                      '-march=icelake-client',
                      ],
                     dependencies : [],
                     link_whole : [],
                     )
