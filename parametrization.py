import time
from z3 import Solver, Int, Array, IntSort, And, Not, If, Select, Store, sat, set_param, get_param, parse_smt2_file
from ConfigSpace import UniformIntegerHyperparameter, Categorical, ConfigurationSpace, Constant
from ConfigSpace.conditions import InCondition
from smac import Scenario
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade
from smac.initial_design.default_design import DefaultInitialDesign

global best_time
global best_config
global categories
global int_categories

best_time = None
best_config = None
categories = {}
int_categories = [ 7 * (10 ** i) for i in range(9) ]

file = open("benches", "r")
lower = 10000
upper = 50000
benchmarks = list(file.readlines())
# benchmarks = benchmarks[:4] + benchmarks[7:]
file.close()

res = []

print(len(benchmarks))
def train(config: ConfigurationSpace, seed):
    global best_time
    global best_config
    global benchmarks
    global res

    times = []
    print(len(benchmarks))

    for lines in benchmarks:
        s = Solver()
        s.set("timeout", 600)
        set_param(**config)
        set_param("sat.random_seed", seed)
        set_param("nlsat.seed", seed)
        set_param("fp.spacer.random_seed", seed)
        set_param("smt.random_seed", seed)
        set_param("sls.random_seed", seed)

        benchmark = "QF_ABV/" + lines[:-1]    
        exps = parse_smt2_file(benchmark)
        start = time.time()
        s.assert_exprs(exps)
        s.check()
        times.append(time.time() - start)

    print("SUM")
    r = sum(times)
    print(r)
    if not res:
        print(seed)
        print(config)
    res.append(r)

    for k in config:
        if config[k] == True:
            categories[k][True].append(r)
        elif config[k] == False:
            categories[k][False].append(r)
        else:
            for ic in int_categories:
                if ic >= config[k]:
                    categories[k][ic].append(r)
                    break

    print(categories)

    return r

cs = ConfigurationSpace()

integer_hyperparameters = [
    ["asymm_branch.delay", 1],
    ["asymm_branch.limit", 100000000],
    ["asymm_branch.rounds", 2],
    ["backtrack.conflicts", 4000],
    ["backtrack.scopes", 100],
    ["burst_search", 100],
    ["probing_cache_limit", 1024],
    ["probing_limit", 5000000],
    ["search.sat.conflicts", 400],
    ["search.unsat.conflicts", 400],
    ["simplify.delay", 0],
    ["subsumption.limit", 100000000],
    ["variable_decay", 110],
    ["lookahead.cube.depth", 1],
]

boolean_hyperparameters = [
    ["abce", False],
    ["acce", False],
    ["anf", False],
    ["anf.exlin", False],
    ["asymm_branch", True],
    ["asymm_branch.sampled", True],
    ["asymm_branch.all", False],
    ["ate", True],
    ["bca", False],
    ["bce", False],
    ["cardinality.solver", True],
    ["core.minimize", False],
    ["core.minimize_partial", False],
    ["cut", False],
    ["cut.aig", False],
    ["cut.dont_cares", True],
    ["cut.force", False],
    ["cut.lut", False],
    ["cut.npn3", True],
    ["cut.redundancies", True],
    ["cut.xor", False],
    ["dyn_sub_res", True],
    ["elim_vars", True],
    ["elim_vars_bdd", True],
    ["enable_pre_simplify", False],
    ["local_search", False],
    ["lookahead.double", True],
    ["lookahead.global_autarky", False],
    ["minimize_lemmas", True],
    ["override_incremental", False],
    ["prob_search", False],
    ["probing", True],
    ["probing_binary", True],
    ["probing_cache", True],
    ["propagate.prefetch", True],
    ["retain_blocked_clauses", True],
    ["subsumption", True]
]

int_upper = 700000000

for param in integer_hyperparameters:
    cs.add_hyperparameter(UniformIntegerHyperparameter(f"sat.{param[0]}", lower=0, upper=int_upper, default_value=param[1]))
    categories[f"sat.{param[0]}"] = {}
    for ic in int_categories:
        categories[f"sat.{param[0]}"][ic] = []

for param in boolean_hyperparameters:
    cs.add_hyperparameter(Categorical(name=f"sat.{param[0]}", items=[True, False], default=param[1]))
    categories[f"sat.{param[0]}"] = { True : [], False : [] }

conditionals = [
    InCondition(child=Categorical(name="sat.asymm_branch.all",items=[True, False], default=False),
                parent=cs["sat.asymm_branch"],
                values=[True]),
    InCondition(child=Categorical(name="sat.asymm_branch.sampled", items=[True, False], default=True),
                parent=cs["sat.asymm_branch"],
                values=[True]),
    InCondition(child=UniformIntegerHyperparameter(name="sat.asymm_branch.delay", lower=0, upper=int_upper, default_value=1),
                parent=cs["sat.asymm_branch"],
                values=[True]),
    InCondition(child=UniformIntegerHyperparameter(name="sat.asymm_branch.limit", lower=0, upper=int_upper, default_value=100000000),
                parent=cs["sat.asymm_branch"],
                values=[True]),
    InCondition(child=UniformIntegerHyperparameter(name="sat.asymm_branch.rounds", lower=0, upper=int_upper, default_value=2),
                parent=cs["sat.asymm_branch"],
                values=[True]),
]

for param in conditionals:
    cs.add_condition(param)

print(cs)

scenario = Scenario(cs, deterministic=True, n_trials=1000, use_default_config=True)

smac = AlgorithmConfigurationFacade(scenario, train, initial_design=DefaultInitialDesign(scenario=scenario), overwrite=False)
print(smac.runhistory.finished)
print(smac.optimize())

print(categories)

for k, v in categories.items():
    vv = []
    for i in v:
        if len(v[i]) != 0:
            vv.append([i, sum(v[i]) / len(v[i])])
    print(k, vv, sep=" ")

print(res)
