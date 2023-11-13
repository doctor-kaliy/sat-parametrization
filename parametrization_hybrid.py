import numpy as np
import subprocess
import os
import time
import sys

seed = int(input())

np.random.seed(seed)

f_counter = 0

def get_c(n):
    items = [i + 1 for i in range(n)]
    probabilities = [1 / i ** 1.5 for i in items]
    probabilities /= np.sum(probabilities)
    return np.random.choice(items, p=probabilities)

def mutate(config, param_name, param_domain):
    for value in param_domain:
        new_config = config.copy()
        new_config[param_name] = value
        yield new_config

cache = {}

def bool_list_to_int(bools):
    return int(''.join(map(str, map(int, bools))), 2)

def call_train(train, mode, config, seed):
    global cache
    global f_counter
    key = tuple(list([v for k, v in config.items()]))
    if key in cache:
        return cache[key]
    value = train(config, mode, seed)
    f_counter += 1
    # TODO change
    cache[key] = value
    return value

def optimize(initial_config, mode, param_domain, steps, train, seed):
    global f_counter

    config = initial_config
    value = call_train(train, mode, config, seed)

    n = len(config)
    c = 1

    no_updates = 0

    for step in range(steps):
        tmp_config = config
        tmp_value = None
        for param_name in param_domain.keys():
            if np.random.rand() < 2 / n:
                for new_config in mutate(tmp_config, param_name, param_domain[param_name]):
                    new_value = call_train(train, mode, new_config, seed)
                    if tmp_value == None or new_value < tmp_value:
                        tmp_value = new_value
                        tmp_config = new_config

        if tmp_value != None and tmp_value < value:
            no_updates = 0
            value = tmp_value
            config = tmp_config
        elif tmp_value != None:
            no_updates += 1

        if no_updates >= 1000:
            config = initial_config
            value = call_train(train, mode, config, seed)
            no_updates = 0

        if step % 10 == 0 or f_counter >= 500:
            print("")
            print("Current step:", step, sep=" ")
            print(config)
            print(value)
            print("")
            sys.stdout.flush() 
            if f_counter >= 500:
                return (config, value)
    return (config, value)

file = open("benches_abv_25_train", "r")
benchmarks = list(file.readlines())
file.close()

res_times = []
res_conflcits = []

param_domain = {}

default_parameters = {
    "smt.arith.nl.grobner": True,
    "smt.arith.auto_config_simplex": False,
    "smt.arith.bprop_on_pivoted_rows": True,
    "smt.arith.dump_lemmas": False,
    "smt.arith.eager_eq_axioms": True,
    "smt.arith.enable_hnf": True,
    "smt.arith.greatest_error_pivot": False,
    "smt.arith.ignore_int": False,
    "smt.arith.int_eq_branch": False,
    "smt.arith.min": False,
    "smt.arith.nl": True,
    "smt.arith.nl.branching": True,
    "smt.arith.nl.expp": False,
    "smt.arith.nl.horner": True,
    "smt.arith.nl.nra": True,
    "smt.arith.nl.order": True,
    "smt.arith.nl.tangents": True,
    "smt.arith.print_ext_var_names": False,
    "smt.arith.print_stats": False,
    "smt.arith.propagate_eqs": True,
    "smt.arith.random_initial_value": False,
    "smt.array.extensional": True,
    "smt.array.weak": False,
    "smt.bv.delay": False,
    "smt.bv.enable_int2bv": True,
    "smt.bv.reflect": True,
    "smt.bv.size_reduce": False,
    "smt.bv.watch_diseq": False,
    "smt.candidate_models": False,
    "smt.clause_proof": False,
    "smt.core.extend_nonlocal_patterns": False,
    "smt.core.extend_patterns": False,
    "smt.core.minimize": False,
    "smt.core.validate": False,
    "smt.elim_unconstrained": True,
    "smt.ematching": True,
    "smt.induction": False,
    "smt.macro_finder": False,
    "smt.mbqi": True,
    "smt.mbqi.trace": False,
    "smt.pb.learn_complements": True,
    "smt.propagate_values": True,
    "smt.pull_nested_quantifiers": False,
    "smt.q.lite": False,
    "smt.quasi_macros": False,
    "smt.restricted_quasi_macros": False,
    "smt.seq.split_w_len": True,
    "smt.seq.validate": False,
    "smt.solve_eqs": True,
    "smt.theory_aware_branching": False,
    "smt.theory_case_split": False,
    'sat.abce': False,
    'sat.acce': False,
    'sat.anf': False,
    'sat.anf.exlin': False,
    'sat.asymm_branch': True,
    'sat.asymm_branch.all': False,
    'sat.bca': False,
    'sat.bce': False,
    'sat.binspr': False,
    'sat.branching.anti_exploration': False,
    'sat.cardinality.solver': True,
    'sat.cce': False,
    'sat.core.minimize': False,
    'sat.core.minimize_partial': False,
    'sat.cut': False,
    'sat.cut.aig': False,
    'sat.cut.dont_cares': True,
    'sat.cut.force': False,
    'sat.cut.lut': False,
    'sat.cut.npn3': False,
    'sat.cut.redundancies': True,
    'sat.enable_pre_simplify': False,
    'sat.elim_vars': True,
    'sat.elim_vars_bdd': True,
    'sat.lookahead.preselect': False,
    'sat.lookahead.use_learned': False,
    'sat.lookahead_scores': False,
    'sat.lookahead_simplify': False,
    'sat.lookahead_simplify.bca': True,
    'sat.local_search': False,
    'sat.local_search_dbg_flips': False,
    'sat.override_incremental': False,
    'sat.phase.sticky': True,
    'sat.prob_search': False,
    'sat.probing': True,
    'sat.probing_binary': True,
    'sat.probing_cache': True,
    'sat.propagate.prefetch': True,
    'sat.scc': True,
    'sat.scc.tr': True,
    'sat.subsumption': True,
}

for k in default_parameters.keys():
    param_domain[k] = [True, False]

integer_hyperparameters = {
    "smt.arith.branch_cut_ratio": 2,
    "smt.arith.nl.delay": 500,
    "smt.arith.nl.grobner_cnfl_to_report": 1,
    "smt.arith.nl.grobner_eqs_growth": 10,
    "smt.arith.nl.grobner_expr_degree_growth": 2,
    "smt.arith.nl.grobner_expr_size_growth": 2,
    "smt.arith.nl.grobner_frequency": 4,
    "smt.arith.nl.grobner_max_simplified": 10000,
    "smt.arith.nl.grobner_subs_fixed": 1,
    "smt.arith.nl.horner_frequency": 4,
    "smt.arith.nl.horner_subs_fixed": 2,
    "smt.arith.nl.rounds": 1024,
    "smt.mbqi.force_template": 10,
    "smt.mbqi.max_cexs": 1,
    "smt.mbqi.max_cexs_incr": 0,
    "smt.mbqi.max_iterations": 1000,
    "smt.pb.conflict_frequency": 1000,
    "smt.qi.max_instances": 4294967295,
    "smt.qi.max_multi_patterns": 0,
    "smt.qi.profile_freq": 4294967295,
    "smt.threads.cube_frequency": 2,
    "smt.threads.max_conflicts": 400,
    'sat.anf.delay': 2,
    'sat.asymm_branch.delay': 1,
    'sat.asymm_branch.rounds': 2,
    'sat.backtrack.conflicts': 4000,
    'sat.backtrack.scopes': 100,
    'sat.bce_at': 2,
    'sat.bce_delay': 2,
    'sat.blocked_clause_limit': 100000000,
    'sat.burst_search': 100,
    'sat.ddfw.init_clause_weight': 8,
    'sat.ddfw.reinit_base': 10000,
    'sat.ddfw.restart_base': 100000,
    'sat.ddfw.use_reward_pct': 15,
    'sat.inprocess.max': 4294967295,
    'sat.lookahead.cube.depth': 1,
    'sat.lookahead.cube.psat.clause_base': 2,
    'sat.lookahead.cube.psat.trigger': 5,
    'sat.lookahead.cube.psat.var_exp': 1,
    'sat.restart.initial': 2,
    'sat.resolution.cls_cutoff1': 100000000,
    'sat.resolution.cls_cutoff2': 700000000,
    'sat.resolution.lit_cutoff_range1': 700,
    'sat.resolution.lit_cutoff_range2': 400,
    'sat.resolution.lit_cutoff_range3': 300,
    'sat.resolution.occ_cutoff': 10,
    'sat.resolution.occ_cutoff_range1': 8,
    'sat.resolution.occ_cutoff_range2': 5,
    'sat.resolution.occ_cutoff_range3': 3,
    'sat.simplify.delay': 0,
    'sat.variable_decay': 110,
}

int_upper = 700000000
int_defaults = list([v for v in integer_hyperparameters.values()])
int_categories = sorted([0, 1] + list([ 5 * (10 ** i) for i in range(9) ]))

for v in integer_hyperparameters:
    param = v
    default_value = integer_hyperparameters[v]
    param_domain[param] = int_categories + [default_value]
    default_parameters[param] = default_value

default_parameters["smt.arith.solver"] = 6
param_domain["smt.arith.solver"] = list(range(7))

default_parameters["sat.lookahead.cube.cutoff"] = "depth"
param_domain["sat.lookahead.cube.cutoff"] = ["depth", "freevars", "psat", "adaptive_freevars", "adaptive_psat"]

default_parameters["sat.pb.solver"] = "solver"
param_domain["sat.pb.solver"] = ["circuit", "sorting", "totalizer", "binary_merge", "segmented", "solver"]

solver_timeout = 45000

print(len(default_parameters))
print(len(benchmarks))
def train(config, mode, seed):
    global benchmarks
    global res
    config["smt.auto_config"] = True
    l = len(benchmarks)
    n = 3
    result = { "time" : [], "conflicts" : [] }
    sampled_benchmarks = np.random.choice(benchmarks, n, replace=False)
    for lines in sampled_benchmarks:
        benchmark, sat_res = lines[:-1].split(", ")
            
        z3_query = ["z3", "-smt2", "-st", benchmark, "timeout=" + str(solver_timeout)]

        for param in config:
            z3_query.append(param + "=" +(str(config[param]).lower()))
        z3_process = subprocess.run(z3_query, capture_output=True)
        z3_result = z3_process.stdout.splitlines()
        result["conflicts"].append(0)
        result["time"].append(0)
        
        if z3_process.returncode != 0:
            print("ERROR")
            print(" ".join(z3_query))
            print(z3_process.stderr)
            return 1000000000
        is_correct = False
        for line in z3_result:
            line = line.decode("utf-8")
            if line == sat_res or line == sat_res + "\n":
                is_correct = True
            kv = line.split()
            if len(kv) == 2:
                key, value = kv
                if key == ":conflicts":
                    result[key[1:]][-1] = float(value)
                if key == ":total-time":
                    result["time"][-1] = float(value[:-1])
        if is_correct == False:
            return 1000000000
    return sum(result[mode]) / len(result[mode])

mode = input()
optimize(default_parameters, mode, param_domain, 20000, train, seed)