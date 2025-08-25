# Arguments
# TARGET: SO | DLL (default: SO)
# DEBUG:  0 | 1  (default: 0)

TARGET ?= LINUX

CXXFLAGS := -Wall -std=c++20 -fdiagnostics-color=always

# Debug/release selection
ifeq ($(DEBUG),1)
    CXXFLAGS += -DDEBUG -D_GLIBCXX_ASSERTIONS=1 -g -O0
else
    CXXFLAGS += -O3
endif

CXXLIBFLAGS := -I../lib -I../src -I../utils -fvisibility=hidden -fvisibility-inlines-hidden # -D_FILE_OFFSET_BITS=64

NAME := x86simdsort

ifeq ($(findstring CYGWIN,$(shell uname)),CYGWIN)
   CYGWIN := 1
else
   CYGWIN := 0
endif
$(info IsCygwin: $(CYGWIN))

ifeq ($(TARGET),DLL)
    # There ia a bug in gcc when the target is the Windows ABI and a function call another function passing by value avx registers, hence we use clang
    # https://github.com/mingw-w64/mingw-w64/issues/115
#   CXX := x86_64-w64-mingw32-g++
    ifeq ($(CYGWIN),1)
       # There ia a bug in gcc when the target is the Windows ABI and a function call another function passing by value avx registers, hence we use clang
       # https://github.com/mingw-w64/mingw-w64/issues/115
       CXX := x86_64-w64-mingw32-clang++
       $(info Using: clang on Cygwin)
    else
       CXX := clang++ -target x86_64-w64-mingw32
    endif
    SHARED_EXT := dll
    LIBNAME := $(NAME)-c-api
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
    LIBNAME := lib$(NAME)-c-api
    CXXLIBFLAGS += -fPIC
    VERDEP := $(LIBNAME).ver
    LDFLAGS := -Wl,--as-needed -Wl,--no-undefined -Wl,-O1 -shared -fPIC \
               -Wl,-soname,$(LIBNAME).so -Wl,--version-script=$(LIBNAME).ver
    SONAME_FLAG := -Wl,-soname,$(LIBNAME).so
    DLLLINK := $(LIBNAME).$(SHARED_EXT)
endif

$(info Compiler: $(shell $(CXX) --version | head -n1))

# Check if configuration changed, if so, clean
CONFIGFILE := config.cfg
CONFIG := $(CXX)-$(TARGET)-$(if $(DEBUG),DEBUG,RELEASE)

# Source and header files
SRC_HEADERS := $(wildcard ../src/*.h) $(wildcard ../src/*.hpp)
UTIL_HEADERS := $(wildcard ../utils/*.h) $(wildcard ../utils/*.hpp)
LIB_HEADERS := $(wildcard ../lib/*.h) $(wildcard ../lib/*.hpp)
ALL_HEADERS := $(UTIL_HEADERS) $(SRC_HEADERS) $(LIB_HEADERS)

LIB_SOURCES := $(wildcard ../lib/$(NAME)-*.cpp)
LIB_OBJECTS := $(LIB_SOURCES:../lib/$(NAME)-%.cpp=$(NAME)-%.cpp.o)


EXTRADEPS := Makefile $(CONFIGFILE)

.PHONY: clean default all lib api smoke

default : lib api

lib : $(LIBNAME).$(SHARED_EXT)
api : $(NAME)-c-api.h
smoke : smoke.exe

all: lib smoke

# Specializations of object file compilation for various architectures
$(NAME).cpp.o:      CXXFLAGS += $(CXXLIBFLAGS) -DC_API_ONLY
$(NAME)-avx2.cpp.o: CXXFLAGS += $(CXXLIBFLAGS) -march=haswell
$(NAME)-skx.cpp.o:  CXXFLAGS += $(CXXLIBFLAGS) -march=skylake-avx512
$(NAME)-icl.cpp.o:  CXXFLAGS += $(CXXLIBFLAGS) -march=icelake-client
$(NAME)-spr.cpp.o:  CXXFLAGS += $(CXXLIBFLAGS) -march=sapphirerapids
smoke.cpp.o:            CXXFLAGS += -I.

ifdef BOOST_ROOT
smoke.cpp.o: CXXFLAGS += -I$(BOOST_ROOT)
endif

# add a dependency to smoke
smoke.cpp.o : $(NAME)-c-api.h

VPATH=../lib:.

# Configuration file generation
$(CONFIGFILE):
	@echo "$(CONFIG)" | cmp -s - $@ 2>/dev/null || echo "$(CONFIG)" > $@

# Object files (note that CXXFLAGS set above is customized for each target)
%.cpp.o: %.cpp $(ALL_HEADERS) $(EXTRADEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# smoke link
smoke.exe: smoke.cpp.o $(LIBNAME).$(SHARED_EXT) $(NAME)-c-api.h $(EXTRADEPS)
	$(CXX) $< -o $@ $(DLLLINK)

# Auto generated header
$(NAME)-c-api.h: ../lib/c-api-header-gen.h ../lib/c-api.sed $(EXTRADEPS)
	cpp -P $< | sed -f ../lib/c-api.sed > $@

# SO shared library export file
$(LIBNAME).ver: ../lib/c-api-ver.h $(CONFIGFILE)
	cpp -P $< | sed "/^$$/d; s/^c_xss/    c_xss/" > $@

# Shared library (SO or DLL) link
$(LIBNAME).$(SHARED_EXT): $(NAME).cpp.o $(LIB_OBJECTS) $(VERDEP) $(EXTRADEPS)
	$(CXX) -o $@ $(NAME).cpp.o $(LIB_OBJECTS) $(LDFLAGS)


clean:
	rm -f *.o *.so *.dll *.lib *.h *.ver *.exe *.cfg
