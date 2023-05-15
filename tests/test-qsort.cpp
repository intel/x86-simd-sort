#include "test-partial-qsort.hpp"
#include "test-qselect.hpp"
#include "test-qsort.hpp"

using QuickSortTestTypes = testing::Types<uint16_t,
                                          int16_t,
                                          float,
                                          double,
                                          uint32_t,
                                          int32_t,
                                          uint64_t,
                                          int64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(TestPrefix, avx512_sort, QuickSortTestTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(TestPrefix, avx512_select, QuickSortTestTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(TestPrefix,
                               avx512_partial_sort,
                               QuickSortTestTypes);
