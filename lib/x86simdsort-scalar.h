#include <algorithm>
#include <numeric>
#define UNUSED(x) (void)(x)
namespace xss {
namespace scalar {
    /* TODO: handle NAN */
    template <typename T>
    void qsort(T *arr, int64_t arrsize)
    {
        std::sort(arr, arr + arrsize);
    }
    template <typename T>
    void qselect(T *arr, int64_t k, int64_t arrsize, bool hasnan)
    {
        UNUSED(hasnan);
        std::nth_element(arr, arr + k, arr + arrsize);
    }
    template <typename T>
    void partial_qsort(T *arr, int64_t k, int64_t arrsize, bool hasnan)
    {
        UNUSED(hasnan);
        std::partial_sort(arr, arr + k, arr + arrsize);
    }
    template <typename T>
    std::vector<int64_t> argsort(T *arr, int64_t arrsize)
    {
	std::vector<int64_t> arg(arrsize);
	std::iota(arg.begin(), arg.end(), 0);
	std::sort(arg.begin(),
		  arg.end(),
		  [arr](int64_t left, int64_t right) -> bool {
		      return arr[left] < arr[right];
		  });
	return arg;
    }
    template <typename T>
    std::vector<int64_t> argselect(T *arr, int64_t k, int64_t arrsize)
    {
	std::vector<int64_t> arg(arrsize);
	std::iota(arg.begin(), arg.end(), 0);
	std::nth_element(arg.begin(),
			 arg.begin() + k,
		  	 arg.end(),
		  	 [arr](int64_t left, int64_t right) -> bool {
		  	     return arr[left] < arr[right];
		  	 });
        return arg;
    }

} // namespace scalar
} // namespace xss
