CXX		= g++-12
SRCDIR		= ./src
TESTDIR		= ./tests
BENCHDIR	= ./benchmarks
UTILS		= ./utils
SRCS		= $(wildcard $(SRCDIR)/*.hpp)
TESTS		= $(wildcard $(TESTDIR)/*.cpp)
TESTOBJS	= $(patsubst $(TESTDIR)/%.cpp,$(TESTDIR)/%.o,$(TESTS))
CXXFLAGS	+= -I$(SRCDIR) -I$(UTILS)
GTESTCFLAGS	= `pkg-config --cflags gtest`
GTESTLDFLAGS	= `pkg-config --libs gtest`
MARCHFLAG	= -march=sapphirerapids -O3

all : test bench

$(UTILS)/cpuinfo.o : $(UTILS)/cpuinfo.cpp
		$(CXX) $(CXXFLAGS) -c $(UTILS)/cpuinfo.cpp -o $(UTILS)/cpuinfo.o

$(TESTDIR)/%.o : $(TESTDIR)/%.cpp $(SRCS)
		$(CXX) $(CXXFLAGS) $(MARCHFLAG) $(GTESTCFLAGS) -c $< -o $@

test: $(TESTOBJS) $(UTILS)/cpuinfo.o $(SRCS)
		$(CXX) $(TESTOBJS) $(UTILS)/cpuinfo.o $(MARCHFLAG) $(CXXFLAGS) -lgtest_main $(GTESTLDFLAGS) -o testexe

bench: $(BENCHDIR)/main.cpp $(SRCS) $(UTILS)/cpuinfo.o
		$(CXX) $(BENCHDIR)/main.cpp $(CXXFLAGS) $(UTILS)/cpuinfo.o $(MARCHFLAG) -o benchexe

meson:
	meson setup builddir && cd builddir && ninja

clean:
	rm -rf $(TESTDIR)/*.o testexe benchexe builddir
