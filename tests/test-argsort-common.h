#include <algorithm>
#include <gtest/gtest.h>
#include <vector>
#include "cpuinfo.h"
#include "rand_array.h"
#include "avx512-64bit-argsort.hpp"

template <typename T>
std::vector<int64_t> std_argsort(const std::vector<T> &array)
{
    std::vector<int64_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(),
              indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array sizeent
                  return array[left] < array[right];
              });

    return indices;
}

#define EXPECT_UNIQUE(sorted_arg) \
    std::sort(sorted_arg.begin(), sorted_arg.end()); \
    std::vector<int64_t> expected_arg(sorted_arg.size()); \
    std::iota(expected_arg.begin(), expected_arg.end(), 0); \
    EXPECT_EQ(sorted_arg, expected_arg) << "Indices aren't unique. Array size = " << sorted_arg.size();
