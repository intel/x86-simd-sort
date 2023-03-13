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
MARCHFLAG	= -march=icelake-client -O3

all : test bench

$(UTILS)/cpuinfo.o : $(UTILS)/cpuinfo.cpp
		$(CXX) $(CXXFLAGS) -c $(UTILS)/cpuinfo.cpp -o $(UTILS)/cpuinfo.o

$(TESTDIR)/%.o : $(TESTDIR)/%.cpp $(SRCS)
		$(CXX) $(CXXFLAGS) $(MARCHFLAG) $(GTESTCFLAGS) -c $< -o $@

test: $(TESTDIR)/main.cpp $(TESTOBJS) $(UTILS)/cpuinfo.o $(SRCS)
		$(CXX) tests/main.cpp $(TESTOBJS) $(UTILS)/cpuinfo.o $(MARCHFLAG) $(CXXFLAGS) $(GTESTLDFLAGS) -o testexe

bench: $(BENCHDIR)/main.cpp $(SRCS) $(UTILS)/cpuinfo.o
		$(CXX) $(BENCHDIR)/main.cpp $(CXXFLAGS) $(UTILS)/cpuinfo.o $(MARCHFLAG) -o benchexe

clean:
		rm -f $(TESTDIR)/*.o testexe benchexe
