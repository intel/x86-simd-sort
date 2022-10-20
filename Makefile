CXX		?= g++
SRCDIR		= ./src
TESTDIR		= ./tests
BENCHDIR	= ./benchmarks
UTILS		= ./utils
SRCS		= $(wildcard $(SRCDIR)/*.hpp)
TESTS		= $(wildcard $(TESTDIR)/*.cpp)
TESTOBJS	= $(patsubst $(TESTDIR)/%.cpp,$(TESTDIR)/%.o,$(TESTS))
TESTOBJS	:= $(filter-out $(TESTDIR)/main.o ,$(TESTOBJS))
GTEST_LIB	= gtest
GTEST_INCLUDE	= /usr/local/include
CXXFLAGS	+= -I$(SRCDIR) -I$(GTEST_INCLUDE) -I$(UTILS)
LD_FLAGS	= -L /usr/local/lib -l $(GTEST_LIB) -l pthread

all : test bench

$(TESTDIR)/%.o : $(TESTDIR)/%.cpp $(SRCS)
		$(CXX) -march=icelake-client -O3 $(CXXFLAGS) -c $< -o $@

test: $(TESTDIR)/main.cpp $(TESTOBJS) $(SRCS)
		$(CXX) tests/main.cpp $(TESTOBJS) $(CXXFLAGS) $(LD_FLAGS) -o testexe

bench: $(BENCHDIR)/main.cpp $(SRCS)
		$(CXX) $(BENCHDIR)/main.cpp $(CXXFLAGS) -march=icelake-client -O3 -o benchexe

clean:
		rm -f $(TESTDIR)/*.o testexe benchexe
