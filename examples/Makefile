CC=g++-12
CFLAGS=-I../src
EXE = argsort kvsort qsortfp16 qsort16 qsort32 qsort64

default: all
all : $(EXE)

qsortfp16: avx512fp-16bit-qsort.cpp
	     $(CC) -o qsortfp16 -O3 -march=sapphirerapids $(CFLAGS) avx512fp-16bit-qsort.cpp

qsort16: avx512-16bit-qsort.cpp
	     $(CC) -o qsort16 -O3 -march=icelake-client $(CFLAGS) avx512-16bit-qsort.cpp

qsort32: avx512-32bit-qsort.cpp
	     $(CC) -o qsort32 -O3 -march=skylake-avx512 $(CFLAGS) avx512-32bit-qsort.cpp

qsort64: avx512-64bit-qsort.cpp
	     $(CC) -o qsort64 -O3 -march=skylake-avx512 $(CFLAGS) avx512-64bit-qsort.cpp

argsort: avx512-argsort.cpp
	     $(CC) -o argsort -O3 -march=skylake-avx512 $(CFLAGS) avx512-argsort.cpp

kvsort: avx512-kv.cpp
	     $(CC) -o kvsort -O3 -march=skylake-avx512 $(CFLAGS) avx512-kv.cpp

clean:
	$(RM) $(EXE)
