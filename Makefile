meson:
	meson setup -Dbuild_tests=true -Dbuild_benchmarks=true --warnlevel 2 --werror --buildtype release builddir
	cd builddir && ninja

mesondebug:
	meson setup -Dbuild_tests=true -Dbuild_benchmarks=true --warnlevel 2 --werror --buildtype debug debug
	cd debug && ninja

clean:
	$(RM) -rf $(TESTOBJS) $(BENCHOBJS) $(UTILOBJS) testexe benchexe builddir debug
