#include "test-argsort-common.h"
#include "test-argsort.hpp"
#include "test-argselect.hpp"

using ArgTestTypes
        = testing::Types<int32_t, uint32_t, float, uint64_t, int64_t, double>;

INSTANTIATE_TYPED_TEST_SUITE_P(T, avx512argsort, ArgTestTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(T, avx512argselect, ArgTestTypes);
