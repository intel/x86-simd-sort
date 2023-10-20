CXX ?= g++-12
CFLAGS = -I../src -std=c++17 -O3
EXE = qsort32avx2 argsort kvsort qsortfp16 qsort16 qsort32 qsort64

default: all
all : $(EXE)

qsortfp16: avx512fp-16bit-qsort.cpp
	     $(CXX) -o qsortfp16 -march=sapphirerapids $(CFLAGS) avx512fp-16bit-qsort.cpp

qsort16: avx512-16bit-qsort.cpp
	     $(CXX) -o qsort16 -march=icelake-client $(CFLAGS) avx512-16bit-qsort.cpp

qsort32: avx512-32bit-qsort.cpp
	     $(CXX) -o qsort32 -march=skylake-avx512 $(CFLAGS) avx512-32bit-qsort.cpp

qsort32avx2: avx2-32bit-qsort.cpp
	     $(CXX) -o qsort32avx2 -march=haswell $(CFLAGS) avx2-32bit-qsort.cpp

qsort64: avx512-64bit-qsort.cpp
	     $(CXX) -o qsort64 -march=skylake-avx512 $(CFLAGS) avx512-64bit-qsort.cpp

argsort: avx512-argsort.cpp
	     $(CXX) -o argsort -march=skylake-avx512 $(CFLAGS) avx512-argsort.cpp

kvsort: avx512-kv.cpp
	     $(CXX) -o kvsort -march=skylake-avx512 $(CFLAGS) avx512-kv.cpp

clean:
	$(RM) $(EXE)
