# When unset, discover g++. Prioritise the latest version on the path.
ifeq (, $(and $(strip $(CXX)), $(filter-out default undefined, $(origin CXX))))
  override CXX	:= $(shell basename `which g++-12 g++-11 g++-10 g++-9 g++-8 g++ | head -n 1`)
endif

export CXX
CXXFLAGS	+= $(OPTIMFLAG) $(MARCHFLAG)
override CXXFLAGS += -I$(SRCDIR) -I$(UTILSDIR)
GTESTCFLAGS	:= `pkg-config --cflags gtest_main`
GTESTLDFLAGS	:= `pkg-config --static --libs gtest_main`
GBENCHCFLAGS	:= `pkg-config --cflags benchmark`
GBENCHLDFLAGS	:= `pkg-config --static --libs benchmark`
OPTIMFLAG	:= -O3
MARCHFLAG	:= -march=sapphirerapids

SRCDIR		:= ./src
TESTDIR		:= ./tests
BENCHDIR	:= ./benchmarks
UTILSDIR	:= ./utils

SRCS		:= $(wildcard $(addprefix $(SRCDIR)/, *.hpp *.h))
UTILSRCS	:= $(wildcard $(addprefix $(UTILSDIR)/, *.hpp *.h))
TESTSRCS	:= $(wildcard $(addprefix $(TESTDIR)/, *.hpp *.h))
BENCHSRCS	:= $(wildcard $(addprefix $(BENCHDIR)/, *.hpp *.h))
UTILS		:= $(wildcard $(UTILSDIR)/*.cpp)
TESTS		:= $(wildcard $(TESTDIR)/*.cpp)
BENCHS		:= $(wildcard $(BENCHDIR)/*.cpp)

test_cxx_flag	= $(shell 2>/dev/null $(CXX) -o /dev/null $(1) -c -x c++ /dev/null; echo $$?)

# Compiling AVX512-FP16 instructions wasn't possible until GCC 12
ifeq ($(call test_cxx_flag,-mavx512fp16), 1)
  BENCHS_SKIP	+= bench-qsortfp16.cpp
  TESTS_SKIP 	+= test-qsortfp16.cpp
endif

# Sapphire Rapids was otherwise supported from GCC 11. Downgrade if required.
ifeq ($(call test_cxx_flag,$(MARCHFLAG)), 1)
  MARCHFLAG	:= -march=icelake-client
endif

BENCHOBJS	:= $(patsubst %.cpp, %.o, $(filter-out $(addprefix $(BENCHDIR)/, $(BENCHS_SKIP)), $(BENCHS)))
TESTOBJS	:= $(patsubst %.cpp, %.o, $(filter-out $(addprefix $(TESTDIR)/, $(TESTS_SKIP)), $(TESTS)))
UTILOBJS	:= $(UTILS:.cpp=.o)

# Stops make from wondering if it needs to generate the .hpp files (.cpp and .h have equivalent rules by default) 
%.hpp:

.PHONY: all
.DEFAULT_GOAL := all
all: test bench

.PHONY: test
test: testexe

.PHONY: bench
bench: benchexe

$(UTILOBJS): $(UTILSRCS)

$(TESTOBJS): $(TESTSRCS) $(UTILSRCS) $(SRCS)
$(TESTDIR)/%.o: override CXXFLAGS += $(GTESTCFLAGS)

testexe: $(TESTOBJS) $(UTILOBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) $(LDFLAGS) -lgtest_main $(GTESTLDFLAGS) -o $@

$(BENCHOBJS): $(BENCHSRCS) $(UTILSRCS) $(SRCS)
$(BENCHDIR)/%.o: override CXXFLAGS += $(GBENCHCFLAGS)

benchexe: $(BENCHOBJS) $(UTILOBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) $(LDFLAGS) -lbenchmark_main $(GBENCHLDFLAGS) -o $@

.PHONY: meson
meson:
	meson setup --warnlevel 0 --buildtype plain builddir
	cd builddir && ninja

.PHONY: clean
clean:
	$(RM) -rf $(TESTOBJS) $(BENCHOBJS) $(UTILOBJS) testexe benchexe builddir
