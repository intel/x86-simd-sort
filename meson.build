project('x86-simd-sort', 'cpp',
        version : '1.0.0',
        license : 'BSD 3-clause')
cpp = meson.get_compiler('cpp')
src = include_directories('src')
bench = include_directories('benchmarks')
utils = include_directories('utils')
tests = include_directories('tests')
gtest_dep = dependency('gtest_main', required : true)
subdir('utils')
subdir('tests')

testexe = executable('testexe',
                     include_directories : [src, utils],
                     dependencies : gtest_dep,
                     link_whole : [libtests, libcpuinfo]
                    )
