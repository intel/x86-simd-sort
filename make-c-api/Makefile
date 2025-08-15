# Arguments
# TARGET: LINUX | WINDOWS (default: LINUX)
# DEBUG:  0 | 1  (default: 0)

TARGET ?= LINUX

CXXCOMMONFLAGS := -Wall -std=c++20 -fdiagnostics-color=always

# Debug/release selection
ifeq ($(DEBUG),1)
    CXXCOMMONFLAGS += -DDEBUG -D_GLIBCXX_ASSERTIONS=1 -g -O0
else
    CXXCOMMONFLAGS += -O3 -g -D_GLIBCXX_ASSERTIONS=1
endif

CXXLIBFLAGS := $(CXXCOMMONFLAGS) -I../lib -I../src -I../utils -fvisibility=hidden -fvisibility-inlines-hidden -D_FILE_OFFSET_BITS=64

NAME := x86simdsort

ifeq ($(findstring CYGWIN,$(shell uname)),CYGWIN)
   CYGWIN := 1
else
   CYGWIN := 0
endif
$(info IsCygwin: $(CYGWIN))

ifeq ($(TARGET),WINDOWS)
    # There ia a bug in gcc when the target is the Windows ABI and a function call another function passing by value avx registers, hence we use clang
    # https://github.com/mingw-w64/mingw-w64/issues/115
#   CXX := x86_64-w64-mingw32-g++
    CXX := x86_64-w64-mingw32-clang++
    SHARED_EXT := dll
    LIBNAME := $(NAME)
#    CXXFLAGS +=
    DLLLINK := $(LIBNAME).$(SHARED_EXT).lib
    STATIC := -static-libstdc++ -static -lpthread -static -pthread
    LDFLAGS := -shared -static-libgcc $(STATIC) -Wl,--out-implib,$(DLLLINK)
#    LDFLAGS += -fsanitize=address,undefined
    SONAME_FLAG :=
else
    ifeq ($(CYGWIN),1)
       # There ia a bug in gcc when the target is the Windows ABI and a function call another function passing by value avx registers, hence we use clang
       # https://github.com/mingw-w64/mingw-w64/issues/115
       CXX := clang++
       $(info Using: clang on Cygwin)
    else
       CXX := g++
    endif
    SHARED_EXT := so
    LIBNAME := lib$(NAME)
    CXXFLAGS += -fPIC
    VERDEP := $(LIBNAME).ver
    LDFLAGS := -Wl,--as-needed -Wl,--no-undefined -Wl,-O1 -shared -fPIC \
               -Wl,-soname,$(LIBNAME).so -Wl,--version-script=$(LIBNAME).ver
    SONAME_FLAG := -Wl,-soname,$(LIBNAME).so
    DLLLINK := $(LIBNAME).$(SHARED_EXT)
endif

#CXXFLAGS = -fvisibility=hidden -fvisibility-inlines-hidden -fdiagnostics-color=always \
#           -D_GLIBCXX_ASSERTIONS=1 -D_FILE_OFFSET_BITS=64 -Wall -Winvalid-pch -std=c++17 -O3 -fPIC

SRC_HEADERS := $(wildcard ../src/*.h) $(wildcard ../src/*.hpp)
UTIL_HEADERS := $(wildcard ../utils/*.h) $(wildcard ../utils/*.hpp)
LIB_HEADERS := $(wildcard ../lib/*.h) $(wildcard ../lib/*.hpp)
ALL_HEADERS := $(UTIL_HEADERS) $(SRC_HEADERS) $(LIB_HEADERS)

LIB_SOURCES := $(wildcard ../lib/x86simdsort-*.cpp)
# to obtain targets, append o to the filename and remove the path
LIB_OBJECTS := $(LIB_SOURCES:../lib/x86simdsort-%.cpp=x86simdsort-%.cpp.o)

.PHONY: all clean

all: $(LIBNAME).$(SHARED_EXT) smoke.exe

# Specializations of object file compilation for various architectures
x86simdsort.cpp.o:      CXXFLAGS := $(CXXLIBFLAGS) -DC_API_ONLY
x86simdsort-avx2.cpp.o: CXXFLAGS := $(CXXLIBFLAGS) -march=haswell
x86simdsort-skx.cpp.o:  CXXFLAGS := $(CXXLIBFLAGS) -march=skylake-avx512
x86simdsort-icl.cpp.o:  CXXFLAGS := $(CXXLIBFLAGS) -march=icelake-client
x86simdsort-spr.cpp.o:  CXXFLAGS := $(CXXLIBFLAGS) -march=sapphirerapids
smoke.cpp.o:            CXXFLAGS := $(CXXCOMMONFLAGS) -I.

# add a dependency to smoke
smoke.cpp.o : $(NAME)-c-api.h

VPATH=../lib:.

# Object files (note that CXXFLAGS set above is customized for each target)
%.cpp.o: %.cpp $(ALL_HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# smoke link
smoke.exe: smoke.cpp.o $(LIBNAME).$(SHARED_EXT) x86simdsort-c-api.h Makefile
	$(CXX) $< -o $@ $(DLLLINK) $(STATIC)

# Auto generated header
$(NAME)-c-api.h: ../lib/x86simdsort-gen-c-api.h ../lib/mk-c-api.sed Makefile
	cpp -P $< | sed -f ../lib/mk-c-api.sed > $@

# SO shared library export file
$(LIBNAME).ver: ../lib/c-api-ver.h Makefile
	cpp -P $< | sed "/^$$/d; s/^c_xss/    c_xss/" > $@

# Shared library (SO or DLL) link
$(LIBNAME).$(SHARED_EXT): x86simdsort.cpp.o $(LIB_OBJECTS) $(VERDEP) Makefile
	$(CXX) -o $@ x86simdsort.cpp.o $(LIB_OBJECTS) $(LDFLAGS)


clean:
	rm -f *.o *.so *.dll *.lib $(NAME)-c-api.h $(LIBNAME).ver *.exe
