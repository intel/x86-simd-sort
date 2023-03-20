CXX		= g++-12
SRCDIR		= ./src
TESTDIR		= ./tests
BENCHDIR	= ./benchmarks
UTILS		= ./utils
SRCS		= $(wildcard $(SRCDIR)/*.hpp)
TESTS		= $(wildcard $(TESTDIR)/*.cpp)
TESTOBJS	= $(patsubst $(TESTDIR)/%.cpp,$(TESTDIR)/%.o,$(TESTS))
CXXFLAGS	+= -I$(SRCDIR) -I$(UTILS)
GTESTCFLAGS	= `pkg-config --cflags gtest_main`
GTESTLDFLAGS	= `pkg-config --libs gtest_main`
GBENCHFLAGS	= `pkg-config --cflags --libs benchmark`
MARCHFLAG	= -march=sapphirerapids -O3

all : test bench

$(UTILS)/cpuinfo.o : $(UTILS)/cpuinfo.cpp
	$(CXX) $(CXXFLAGS) -c $(UTILS)/cpuinfo.cpp -o $(UTILS)/cpuinfo.o

$(TESTDIR)/%.o : $(TESTDIR)/%.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $(MARCHFLAG) $(GTESTCFLAGS) -c $< -o $@

test: $(TESTOBJS) $(UTILS)/cpuinfo.o $(SRCS)
	$(CXX) $(TESTOBJS) $(UTILS)/cpuinfo.o $(MARCHFLAG) $(CXXFLAGS) -lgtest_main $(GTESTLDFLAGS) -o testexe

bench: $(SRCS) $(UTILS)/cpuinfo.o
	$(CXX) $(MARCHFLAG) $(CXXFLAGS) $(GBENCHFLAGS) $(UTILS)/cpuinfo.o -o benchexe

meson:
	meson setup --warnlevel 0 --buildtype plain builddir
	cd builddir && ninja

clean:
	$(RM) -rf $(TESTDIR)/*.o $(UTILS)/*.o testexe benchexe builddir
