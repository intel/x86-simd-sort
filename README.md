# x86-simd-sort

C++ template library for high performance SIMD based sorting routines for
16-bit, 32-bit and 64-bit data types. The sorting routines are accelerated
using AVX-512/AVX2 when available. The library auto picks the best version
depending on the processor it is run on. If you are looking for the AVX-512 or
AVX2 specific implementations, please see
[README](https://github.com/intel/x86-simd-sort/blob/main/src/README.md) file under
`src/` directory. The following routines are currently supported:


### Sort routines on arrays
```cpp
x86simdsort::qsort(T* arr, size_t size, bool hasnan);
x86simdsort::qselect(T* arr, size_t k, size_t size, bool hasnan);
x86simdsort::partial_qsort(T* arr, size_t k, size_t size, bool hasnan);
```
Supported datatypes: `T` $\in$ `[_Float16, uint16_t, int16_t, float, uint32_t,
int32_t, double, uint64_t, int64_t]`

### Key-value sort routines on pairs of arrays
```cpp
x86simdsort::keyvalue_qsort(T1* key, T2* val, size_t size, bool hasnan);
```
Supported datatypes: `T1`, `T2` $\in$ `[float, uint32_t, int32_t, double,
uint64_t, int64_t]` Note that keyvalue sort is not yet supported for 16-bit
data types.

### Arg sort routines on arrays
```cpp
std::vector<size_t> arg = x86simdsort::argsort(T* arr, size_t size, bool hasnan);
std::vector<size_t> arg = x86simdsort::argselect(T* arr, size_t k, size_t size, bool hasnan);
```
Supported datatypes: `T` $\in$ `[_Float16, uint16_t, int16_t, float, uint32_t,
int32_t, double, uint64_t, int64_t]`

## Build/Install

[meson](https://github.com/mesonbuild/meson) is the used build system. Command
to build and install the library:

```
meson setup --buildtype release builddir && cd builddir
meson compile
sudo meson install
```

Once installed, you can use `pkg-config --cflags --libs x86simdsortcpp` to
populate the right cflags and ldflags to compile and link your C++ program.
This repository also contains a test suite and benchmarking suite which are
written using [googletest](https://github.com/google/googletest) and [google
benchmark](https://github.com/google/benchmark) frameworks respectively. You
can configure meson to build them both by using `-Dbuild_tests=true` and
`-Dbuild_benchmarks=true`.

## Example usage

```cpp
#include "x86simdsort.h"

int main() {
    std::vector<float> arr{1000};
    x86simdsort::qsort(arr, 1000, true);
    return 0;
}
```


## Details

- `x86simdsort::qsort` is equivalent to `qsort` in
  [C](https://www.tutorialspoint.com/c_standard_library/c_function_qsort.htm)
  or `std::sort` in [C++](https://en.cppreference.com/w/cpp/algorithm/sort).
- `x86simdsort::qselect` is equivalent to `std::nth_element` in
  [C++](https://en.cppreference.com/w/cpp/algorithm/nth_element) or
  `np.partition` in
  [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.partition.html).
- `x86simdsort::partial_qsort` is equivalent to `std::partial_sort` in
  [C++](https://en.cppreference.com/w/cpp/algorithm/partial_sort).
- `x86simdsort::argsort` is equivalent to `np.argsort` in
  [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html).
- `x86simdsort::argselect` is equivalent to `np.argpartition` in
  [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html).

Supported datatypes: `uint16_t, int16_t, _Float16, uint32_t, int32_t, float,
uint64_t, int64_t, double`. Note that `_Float16` will require building this
library with g++ >= 12.x. All the functions have an optional argument `bool
hasnan` set to `false` by default (these are relevant to floating point data
types only).  If your array has NAN's, the the behaviour of the sorting routine
is undefined. If `hasnan` is set to true, NAN's are always sorted to the end of
the array. In addition to that, qsort will replace all your NAN's with
`std::numeric_limits<T>::quiet_NaN`. The original bit-exact NaNs in
the input are not preserved. Also note that the arg methods (argsort and
argselect) will not use the SIMD based algorithms if they detect NAN's in the
array. You can read details of all the implementations
[here](https://github.com/intel/x86-simd-sort/src/README.md).

## Downstream projects using x86-simd-sort

- NumPy uses this as a [submodule](https://github.com/numpy/numpy/pull/22315) to accelerate `np.sort, np.argsort, np.partition and np.argpartition`.
- A slightly modifed version this library has been integrated into [openJDK](https://github.com/openjdk/jdk/pull/14227).
- [GRAPE](https://github.com/alibaba/libgrape-lite.git): C++ library for parallel graph processing.
- AVX-512 version of the key-value sort has been submitted to [Oceanbase](https://github.com/oceanbase/oceanbase/pull/1325).
