template <typename T>
std::vector<int64_t> stdargsort(const std::vector<T> &array)
{
    std::vector<int64_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(),
              indices.end(),
              [&array](int64_t left, int64_t right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}

template <typename T, class... Args>
static void scalarargsort(benchmark::State &state, Args &&...args)
{
    // get args
    auto args_tuple = std::make_tuple(std::move(args)...);
    size_t arrsize = std::get<0>(args_tuple);
    std::string arrtype = std::get<1>(args_tuple);
    // set up array
    std::vector<T> arr = get_array<T>(arrtype, arrsize);
    std::vector<T> arr_bkp = arr;
    std::vector<int64_t> inx;
    // benchmark
    for (auto _ : state) {
        inx = stdargsort(arr);
    }
}

template <typename T, class... Args>
static void simdargsort(benchmark::State &state, Args &&...args)
{
    // get args
    auto args_tuple = std::make_tuple(std::move(args)...);
    size_t arrsize = std::get<0>(args_tuple);
    std::string arrtype = std::get<1>(args_tuple);
    // set up array
    std::vector<T> arr = get_array<T>(arrtype, arrsize);
    std::vector<T> arr_bkp = arr;
    std::vector<int64_t> inx;
    // benchmark
    for (auto _ : state) {
        inx = x86simdsort::argsort(arr.data(), arrsize);
    }
}

#define BENCH_BOTH(type) \
    BENCH_SORT(simdargsort, type) \
    BENCH_SORT(scalarargsort, type)

BENCH_BOTH(int64_t)
BENCH_BOTH(uint64_t)
BENCH_BOTH(double)
BENCH_BOTH(int32_t)
BENCH_BOTH(uint32_t)
BENCH_BOTH(float)
