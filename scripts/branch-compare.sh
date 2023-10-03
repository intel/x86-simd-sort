#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname $SCRIPT_DIR)
branch=$(git rev-parse --abbrev-ref HEAD)
#br_commit=$(git rev-parse $branch)
#main_commit=$(git rev-parse main)
echo "Comparing main branch with $branch"

build_branch() {
    dir_name=$1
    if [ ! -d $dir_name ]; then
        git clone -b $dir_name ${BASE_DIR} $dir_name
    else
        # if it exists, just update it
        cd $dir_name
        git fetch origin
        git rebase origin/$dir_name
        # rebase fails with conflict, delete and start over
        if [ "$?" != 0 ]; then
            cd ..
            rm -rf $dir_name
            git clone -b $dir_name ${BASE_DIR} $dir_name
        else
            cd ..
        fi
    fi
    cd $dir_name
    meson setup --warnlevel 0 --buildtype release builddir
    cd builddir
    ninja
    cd ../../
}

mkdir -p .bench
cd .bench
if [ ! -d google-benchmark ]; then
    git clone https://github.com/google/benchmark google-benchmark
fi
compare=$(realpath google-benchmark/tools/compare.py)
build_branch $branch
build_branch "main"
contender=$(realpath ${branch}/builddir/benchexe)
baseline=$(realpath main/builddir/benchexe)

if [ -z "$1" ]; then
    echo "Comparing all benchmarks .."
    $compare benchmarks $baseline $contender
else
    echo "Comparing benchmark $1 .."
    $compare benchmarksfiltered $baseline $1 $contender $1
fi
