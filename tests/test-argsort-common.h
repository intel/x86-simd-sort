#include "avx512-64bit-argsort.hpp"

#include "rand_array.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

template <typename T>
std::vector<int64_t> std_argsort(const std::vector<T> &arr)
{
    std::vector<int64_t> indices(arr.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(),
              indices.end(),
              [&arr](int64_t left, int64_t right) -> bool {
                  if ((!std::isnan(arr[left])) && (!std::isnan(arr[right]))) {
                      return arr[left] < arr[right];
                  }
                  else if (std::isnan(arr[left])) {
                      return false;
                  }
                  else {
                      return true;
                  }
              });

    return indices;
}

template <typename T>
T std_min_element(std::vector<T> arr,
                  std::vector<int64_t> arg,
                  int64_t left,
                  int64_t right)
{
    std::vector<int64_t>::iterator res = std::min_element(
            arg.begin() + left,
            arg.begin() + right,
            [arr](int64_t a, int64_t b) -> bool {
                if ((!std::isnan(arr[a])) && (!std::isnan(arr[b]))) {
                    return arr[a] < arr[b];
                }
                else if (std::isnan(arr[a])) {
                    return false;
                }
                else {
                    return true;
                }
            });
    return arr[*res];
}

template <typename T>
T std_max_element(std::vector<T> arr,
                  std::vector<int64_t> arg,
                  int64_t left,
                  int64_t right)
{
    std::vector<int64_t>::iterator res = std::max_element(
            arg.begin() + left,
            arg.begin() + right,
            [arr](int64_t a, int64_t b) -> bool {
                if ((!std::isnan(arr[a])) && (!std::isnan(arr[b]))) {
                    return arr[a] > arr[b];
                }
                else if (std::isnan(arr[a])) {
                    return true;
                }
                else {
                    return false;
                }
            });
    return arr[*res];
}

#define EXPECT_UNIQUE(sorted_arg) \
    std::sort(sorted_arg.begin(), sorted_arg.end()); \
    std::vector<int64_t> expected_arg(sorted_arg.size()); \
    std::iota(expected_arg.begin(), expected_arg.end(), 0); \
    EXPECT_EQ(sorted_arg, expected_arg) \
            << "Indices aren't unique. Array size = " << sorted_arg.size();
