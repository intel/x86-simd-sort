project('x86-simd-sort', 'cpp',
        version : '4.0.0',
        license : 'BSD 3-clause',
        default_options : ['cpp_std=c++17'])
cpp = meson.get_compiler('cpp')
src = include_directories('src')
lib = include_directories('lib')
bench = include_directories('benchmarks')
utils = include_directories('utils')
tests = include_directories('tests')

fp16code = '''#include<immintrin.h>
int main() {
  __m512h temp = _mm512_set1_ph(1.0f);
  __m512h var2 = _mm512_min_ph(temp, temp);
  return 0;
}
'''
cancompilefp16 = cpp.compiles(fp16code, args:'-march=sapphirerapids')
flags_hide_symbols = ['-fvisibility=hidden', '-fvisibility-inlines-hidden']

subdir('lib')
libsimdsort = shared_library('x86simdsort',
                             'lib/x86simdsort.cpp',
                             include_directories : [utils, lib],
                             link_with : [libtargets],
                             cpp_args : [flags_hide_symbols],
                             install : true,
                            )

pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries : libsimdsort,
                 version : '4.0',
                 name : 'libx86simdsort',
                 filebase : 'x86simdsort',
                 description : 'High performance SIMD based sorting routines.')

# Build test suite if option build_tests set to true
if get_option('build_tests')
  gtest_dep = dependency('gtest_main', required : true, static: false)
  subdir('tests')
  testexe = executable('testexe',
                     include_directories : [lib, utils],
                     dependencies : gtest_dep,
                     link_whole : [libtests],
                     link_with : libsimdsort,
                    )
  test('x86 simd sort tests', testexe)
endif

# Build benchmarking suite if option build_benchmarks is set to true
if get_option('build_benchmarks')
  gbench_dep = dependency('benchmark', required : true, static: false)
  subdir('benchmarks')
  benchexe = executable('benchexe',
                      include_directories : [src, lib, utils, bench],
                      dependencies : [gbench_dep],
                      link_args: ['-lbenchmark_main'],
                      link_whole : [libbench],
                      link_with : libsimdsort,
                     )
endif

summary({
  'Can compile AVX-512 FP16 ISA': cancompilefp16,
  'Build test content': get_option('build_tests'),
  'Build benchmarks': get_option('build_benchmarks'),
  },
  section: 'Configuration',
  bool_yn: true
  )

