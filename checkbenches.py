import time
import random
import sys
from z3 import Solver, unknown, Int, Array, IntSort, And, Not, If, Select, Store, sat, set_param, get_param, parse_smt2_file

def check_range(s):
    return s >= 20000 and s <= 28000

# benchmarks = list(f.split(", ")[0] for f in file.readlines() if check_range(int(f.split(", ")[1])))
lines = []

file = open("measure_alia_all", "r")
lines = lines + list(file.readlines())
file.close()

file = open("measure_alia_all_1", "r")
lines = lines + list(file.readlines())
file.close()

benchmarks = list(f for f in set(lines) if "," in f and check_range(int(f.split(", ")[1])))

print(len(benchmarks))

import random

random.shuffle(benchmarks)

for line in benchmarks:
    benchmark, time, result = line.split(", ")
    if result == "unknown" or result == "unknown\n":
        continue
    print(benchmark, result[:-1], sep=", ")
    sys.stdout.flush()


# random.shuffle(benchmarks)

# for b in benchmarks[:150]:
#     print(b)