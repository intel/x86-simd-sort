import sys
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--branchcompare', help='Compare benchmarks of current branch with main. Provide an optional --filter')
parser.add_argument('--benchcompare', action='store_true', help='Compare across benchmarks. Requires --baseline and --contender')
parser.add_argument("-f", '--filter', type=str, required=False)
parser.add_argument("-b", '--baseline', type=str, required=False)
parser.add_argument("-c", '--contender', type=str, required=False)
args = parser.parse_args()

if len(sys.argv) == 1:
        parser.error("requires one of --benchcompare or --branchcompare")

if args.benchcompare:
    if (args.baseline is None or args.contender is None):
        parser.error("--benchcompare requires --baseline and --contender")
    else:
        rc = subprocess.check_call("./bench-compare.sh '%s %s'" % (args.baseline, args.contender), shell=True)

if args.branchcompare:
    if args.filter is None:
        rc = subprocess.call("./branch-compare.sh")
    else:
        rc = subprocess.check_call("./branch-compare.sh '%s'" % args.filter, shell=True)
