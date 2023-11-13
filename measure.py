from z3 import Solver, unknown, Int, Array, IntSort, And, Not, If, Select, Store, sat, set_param, get_param, parse_smt2_file
import os
import time

def walk_directory(path, f):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            f(file_path)

def get_benchmarks(path):
    res = []
    walk_directory(path, lambda path : res.append(path))
    return res

benchmarks = get_benchmarks("QF_ALIA")

benchmarks = list([benchmark for benchmark in benchmarks if ".smt2" in benchmark])

lines = []

file = open("measure_alia_all", "r")
lines = lines + list(file.readlines())
file.close()


file = open("measure_alia_all_1", "r")
lines = lines + list(file.readlines())
file.close()

exclude = set([l.split(", ")[0] for l in lines if not (l.split(", ")[2] in set(["unknown\n", "unknown"]))])

seed = 42
import subprocess

def train(config, benchmark, seed):
    z3_query = ["z3", "-smt2", "-st", benchmark, "timeout=35000"]

    for param in config:
        z3_query.append(param + "=" +(str(config[param]).lower()))
    z3_process = subprocess.run(z3_query, capture_output=True)
    z3_result = z3_process.stdout.splitlines()
    
    if z3_process.returncode != 0:
        print("ERROR")
        print(config)
        print(benchmark)
        print(z3_process.stderr)
        exit(0)

    sat_res = None
    for line in z3_result:
        line = line.decode("utf-8")
        if line == "sat" or line == "unknown\n" or line == "unknown" or line == "sat\n" or line == "unsat" or line == "unsat\n":
            sat_res = line
        kv = line.split()
        if len(kv) == 2:
            key, value = kv
            # key = key.decode("utf-8")
            # value = value.decode("utf-8")
            if key == ":total-time":
                return (float(value[:-1]), sat_res)
    return (1000000000.0, "unknown")

import sys
import random

random.shuffle(benchmarks)

for benchmark in benchmarks:
    if benchmark in exclude:
        continue
    sum, res = train({}, benchmark, seed)
    print(benchmark, int((1000 * sum)), res, sep = ", ")
    sys.stdout.flush()
