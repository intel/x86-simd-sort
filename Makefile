CXX		?= g++
SRCDIR		= ./src
TESTDIR		= ./tests
BENCHDIR	= ./benchmarks
UTILS		= ./utils
SRCS		= $(wildcard $(SRCDIR)/*.hpp)
TESTS		= $(wildcard $(TESTDIR)/*.cpp)
TESTOBJS	= $(patsubst $(TESTDIR)/%.cpp,$(TESTDIR)/%.o,$(TESTS))
TESTOBJS	:= $(filter-out $(TESTDIR)/main.o ,$(TESTOBJS))
CXXFLAGS	+= -I$(SRCDIR) -I$(UTILS)
GTESTCFLAGS	= `pkg-config --cflags gtest`
GTESTLDFLAGS	= `pkg-config --libs gtest`

all : test bench

$(TESTDIR)/%.o : $(TESTDIR)/%.cpp $(SRCS)
		$(CXX) -march=icelake-client -O3 $(CXXFLAGS) $(GTESTCFLAGS) -c $< -o $@

test: $(TESTDIR)/main.cpp $(TESTOBJS) $(SRCS)
		$(CXX) tests/main.cpp $(TESTOBJS) $(CXXFLAGS) $(GTESTLDFLAGS) -o testexe

bench: $(BENCHDIR)/main.cpp $(SRCS)
		$(CXX) $(BENCHDIR)/main.cpp $(CXXFLAGS) -march=icelake-client -O3 -o benchexe

clean:
		rm -f $(TESTDIR)/*.o testexe benchexe
