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
#include <string.h>
#include <getopt.h>

void print_usage (int exit_code)
{
    printf("Usage:  %s options [ ... ]\n", "benchexe");
    printf(  "  -h  --help			        Display this usage information.\n"
             "  -d  --dtypesize <list dtypes>		List of dtype sizes in bytes to benchmark. Options: 2,4,8\n"
             "  -a  --arraytype <array data types>	Types of data in random unsorted array. Options: uniform, reverse, ordered, limitedrange\n"
             "  -s  --seed <random generator seed>	Seed for the random number generator as an int\n"
          );
    exit(exit_code);
}

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
void run_bench(const std::string datatype, int seed)
{
    std::streamsize ss = std::cout.precision();
    std::cout << std::fixed;
    std::cout << std::setprecision(1);
    std::vector<int> array_sizes = {10000, 100000, 1000000};
    for (auto size : array_sizes) {
        std::vector<T> arr;
        if (datatype.find("uniform") != std::string::npos) {
            arr = get_uniform_rand_array<T>(size, seed);
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
            arr = get_uniform_rand_array<T>(size, seed, (T)10, (T)0);
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

void bench_all(const std::string datatype,
               std::vector<int> dtypesizes,
               int seed)
{
    if (cpu_has_avx512bw()) {
        if (std::find(dtypesizes.begin(), dtypesizes.end(), 4) != dtypesizes.end()) {
            run_bench<uint32_t>(datatype, seed);
            run_bench<int32_t>(datatype, seed);
            run_bench<float>(datatype, seed);
        }
        if (std::find(dtypesizes.begin(), dtypesizes.end(), 8) != dtypesizes.end()) {
            run_bench<uint64_t>(datatype, seed);
            run_bench<int64_t>(datatype, seed);
            run_bench<double>(datatype, seed);
        }
        if (std::find(dtypesizes.begin(), dtypesizes.end(), 2) != dtypesizes.end()) {
            if (cpu_has_avx512_vbmi2()) {
                run_bench<uint16_t>(datatype, seed);
                run_bench<int16_t>(datatype, seed);
            }
            else {
                printf("CPU doesn't support AVX512 VBMI2, skipping uint16_t and int16_t\n");
            }
        }
    }
    else {
        printf("CPU doesn't support AVX512 BW, cannot benchmark\n");
    }
}


int main(int argc, char *argv[])
{
    int c;
    std::vector<std::string> arraytypes = {"uniform,reverse,ordered,limitedrange"};
    std::vector<int> dtypesizes = {2,4,8};
    int seed = 42;

    while (1) {
        int this_option_optind = optind ? optind : 1;
        int option_index = 0;
        static struct option long_options[] = {
            {"dtypesize",   required_argument, 0,  'd' },
            {"arraytype",   required_argument, 0,  'a' },
            {"seed",        required_argument, 0,  's' },
            {"help",        no_argument,       0,  'h' },
        };
        c = getopt_long(argc, argv, "d:a:s:h", long_options, &option_index);
        if (c == -1)
            break;

       switch (c) {

       case 'd':
            {
                dtypesizes.clear();
                char *token = strtok(optarg, ",");
                while (token != NULL) {
                    dtypesizes.push_back(atoi(token));
                    token = strtok(NULL, ",");
                }
                break;
            }

       case 'a':
            {
                arraytypes.clear();
                char *token = strtok(optarg, ",");
                while (token != NULL) {
                    arraytypes.push_back(token);
                    token = strtok(NULL, ",");
                }
                break;
            }

       case 's':
            seed = atoi(optarg);
            break;

       case 'h':
            print_usage(0);
            break;

       case '?':
            print_usage(1);
            break;

       default:
            printf("?? getopt returned character code 0%o ??\n", c);
        }
    }
    if (optind < argc) {
        printf("Ignored non-option ARGV-elements: ");
        while (optind < argc)
            printf("%s ", argv[optind++]);
        printf("\n");
    }
    printLine(' ',
              "array type",
              "typeid name",
              "dtype size",
              "array size",
              "avx512 sort",
              "std sort",
              "speed up");
    printLine('-', "", "", "", "", "", "", "");
    for (auto type : arraytypes) {
        bench_all(type, dtypesizes, seed);
    }
    printLine('-', "", "", "", "", "", "", "");
    return 0;
}
