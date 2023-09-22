project('x86-simd-sort', 'cpp',
        version : '2.0.0',
        license : 'BSD 3-clause',
        default_options : ['cpp_std=c++17'])
cpp = meson.get_compiler('cpp')
src = include_directories('src')
lib = include_directories('lib')
bench = include_directories('benchmarks')
utils = include_directories('utils')
tests = include_directories('tests')
gtest_dep = dependency('gtest_main', required : true, static: true)
gbench_dep = dependency('benchmark', required : true, static: true)

fp16code = '''#include<immintrin.h>
int main() {
  __m512h temp = _mm512_set1_ph(1.0f);
  __m512h var2 = _mm512_min_ph(temp, temp);
  return 0;
}
'''
cancompilefp16 = cpp.compiles(fp16code, args:'-march=sapphirerapids')

subdir('lib')
subdir('tests')
subdir('benchmarks')

libsimdsort = shared_library('x86simdsort',
                             'lib/x86simdsort.cpp',
                             include_directories : [utils, lib],
                             link_whole : [libtargets],
                             cpp_args : ['-O3'],
                            )

testexe = executable('testexe',
                     include_directories : [lib, utils],
                     dependencies : gtest_dep,
                     link_whole : [libtests],
                     link_with : libsimdsort,
                    )

benchexe = executable('benchexe',
                      include_directories : [src, lib, utils, bench],
                      dependencies : [gbench_dep],
                      link_args: ['-lbenchmark_main'],
                      link_whole : [libbench],
                      link_with : libsimdsort,
                     )

summary({
  'Can compile AVX-512 FP16 ISA': cancompilefp16,
  },
  section: 'Configuration',
  bool_yn: true
  )

