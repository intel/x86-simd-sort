/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "bench.hpp"
#include "cpuinfo.h"
#include "rand_array.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7>
void printLine(const char fill, T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7)
{
    std::cout << std::left << std::setw(3) << std::setfill(fill) << " | ";
    std::cout << std::left << std::setw(15) << std::setfill(fill) << t1
              << " | ";
    std::cout << std::left << std::setw(13) << std::setfill(fill) << t2
              << " | ";
    std::cout << std::left << std::setw(13) << std::setfill(fill) << t3
              << " | ";
    std::cout << std::left << std::setw(13) << std::setfill(fill) << t4
              << " | ";
    std::cout << std::left << std::setw(13) << std::setfill(fill) << t5
              << " | ";
    std::cout << std::left << std::setw(13) << std::setfill(fill) << t6
              << " | ";
    std::cout << std::left << std::setw(13) << std::setfill(fill) << t7
              << " | ";
    std::cout << std::endl;
}

template <typename T>
void run_bench(const std::string datatype)
{
    std::streamsize ss = std::cout.precision();
    std::cout << std::fixed;
    std::cout << std::setprecision(1);
    std::vector<int> array_sizes = {10000, 100000, 1000000};
    for (auto size : array_sizes) {
        std::vector<T> arr;
        if (datatype.find("uniform") != std::string::npos) {
            arr = get_uniform_rand_array<T>(size);
        }
        else if (datatype.find("reverse") != std::string::npos) {
            for (int ii = 0; ii < size; ++ii) {
                arr.emplace_back((T)(size - ii));
            }
        }
        else if (datatype.find("ordered") != std::string::npos) {
            for (int ii = 0; ii < size; ++ii) {
                arr.emplace_back((T)ii);
            }
        }
        else if (datatype.find("limited") != std::string::npos) {
            arr = get_uniform_rand_array<T>(size, (T)10, (T)0);
        }
        else {
            std::cout << "Skipping unrecognized array type: " << datatype
                      << std::endl;
            return;
        }
        auto out = bench_sort(arr, 20, 10);
        printLine(' ',
                  datatype,
                  typeid(T).name(),
                  sizeof(T),
                  size,
                  std::get<0>(out),
                  std::get<1>(out),
                  (float)std::get<1>(out) / std::get<0>(out));
    }
    std::cout << std::setprecision(ss);
}

void bench_all(const std::string datatype)
{
    if (cpu_has_avx512bw()) {
        run_bench<uint32_t>(datatype);
        run_bench<int32_t>(datatype);
        run_bench<float>(datatype);
        run_bench<uint64_t>(datatype);
        run_bench<int64_t>(datatype);
        run_bench<double>(datatype);
        if (cpu_has_avx512_vbmi2()) {
            run_bench<uint16_t>(datatype);
            run_bench<int16_t>(datatype);
        }
    }
}

int main(/*int argc, char *argv[]*/)
{
    printLine(' ',
              "array type",
              "typeid name",
              "dtype size",
              "array size",
              "avx512 sort",
              "std sort",
              "speed up");
    printLine('-', "", "", "", "", "", "", "");
    bench_all("uniform random");
    bench_all("reverse");
    bench_all("ordered");
    bench_all("limitedrange");
    printLine('-', "", "", "", "", "", "", "");
    return 0;
}
