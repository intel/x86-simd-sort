#~/bin/bash
set -e
branch=$(git rev-parse --abbrev-ref HEAD)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..
if [[ -z "${GBENCH}" ]]; then
    echo "Please set env variable GBENCH and re run"
    exit 1
fi

compare=$GBENCH/tools/compare.py
if [ ! -f $compare ]; then
    echo "Unable to locate $GBENCH/tools/compare.py"
    exit 1
fi

meson setup --warnlevel 0 --buildtype plain builddir-${branch}
cd builddir-${branch}
ninja
$compare filters ./benchexe $1 $2
