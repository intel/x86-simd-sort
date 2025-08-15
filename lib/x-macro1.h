
#ifndef XSS_XI1
#	error XSS_XI1 must be defined
#endif

#ifndef XSS_XF1
#	error XSS_XF1 must be defined
#endif

XSS_XI1(uint32, uint32_t)
XSS_XI1(uint64, uint64_t)
XSS_XI1(int64, int64_t)
XSS_XI1(int32, int32_t)
XSS_XF1(float, float)
XSS_XF1(double, double)

#undef XSS_XI1
#undef XSS_XF1
