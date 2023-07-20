#include <algorithm>
#include <gtest/gtest.h>
#include <vector>
#include "cpuinfo.h"
#include "rand_array.h"
#include "avx512-64bit-argsort.hpp"

template <typename T>
std::vector<int64_t> std_argsort(const std::vector<T> &arr)
{
    std::vector<int64_t> indices(arr.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(),
              indices.end(),
              [&arr](int64_t left, int64_t right) -> bool {
                    if ((!std::isnan(arr[left])) && (!std::isnan(arr[right]))) {return arr[left] < arr[right];}
                    else if (std::isnan(arr[left])) {return false;}
                    else {return true;}
                });

    return indices;
}

#define EXPECT_UNIQUE(sorted_arg) \
    std::sort(sorted_arg.begin(), sorted_arg.end()); \
    std::vector<int64_t> expected_arg(sorted_arg.size()); \
    std::iota(expected_arg.begin(), expected_arg.end(), 0); \
    EXPECT_EQ(sorted_arg, expected_arg) << "Indices aren't unique. Array size = " << sorted_arg.size();
