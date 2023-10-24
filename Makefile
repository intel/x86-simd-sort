meson:
	meson setup --warnlevel 2 --werror --buildtype release builddir
	cd builddir && ninja

mesondebug:
	meson setup --warnlevel 2 --werror --buildtype debug debug
	cd debug && ninja

clean:
	$(RM) -rf $(TESTOBJS) $(BENCHOBJS) $(UTILOBJS) testexe benchexe builddir
