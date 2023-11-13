import time
import smac

from z3 import Solver, Int, Array, IntSort, And, Not, If, Select, Store, sat, set_param, get_param, parse_smt2_file
from ConfigSpace import UniformIntegerHyperparameter, Categorical, ConfigurationSpace, Constant
from ConfigSpace.conditions import InCondition
from smac import Scenario
from smac import Callback
from smac.runhistory import TrialInfo, TrialValue
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade
from smac.initial_design.default_design import DefaultInitialDesign

file = open("benches3", "r")
benchmarks = list(file.readlines())
file.close()

res = []

print(len(benchmarks))
def train(config: ConfigurationSpace, seed):
    global benchmarks
    global res

    launches = 3
    r = 0

    for i in range(launches):
        times = []
        for lines in benchmarks:
            s = Solver()
            s.set("timeout", 600)
            set_param(**config)
            set_param("sat.random_seed", seed)
            set_param("nlsat.seed", seed)
            set_param("fp.spacer.random_seed", seed)
            set_param("smt.random_seed", seed)
            set_param("sls.random_seed", seed)

            benchmark = lines[:-1]    
            exps = parse_smt2_file(benchmark)
            start = time.time()
            s.assert_exprs(exps)
            s.check()
            times.append(time.time() - start)

        print("SUM")
        r += sum(times)
    r /= 3
    print(r)
    if not res:
        print(seed)
        print(config)
    res.append(r)

    if r >= 40:
        return 1000000
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

int_defaults = [0, 110, 1, 100000000, 2, 4000, 100, 1024, 5000000, 400]

int_categories = sorted(int_defaults + list([ 7 * (10 ** i) for i in range(9) ]))

for param in integer_hyperparameters:
    cs.add_hyperparameter(Categorical(name=f"sat.{param[0]}", items=int_categories, default=param[1]))

for param in boolean_hyperparameters:
    cs.add_hyperparameter(Categorical(name=f"sat.{param[0]}", items=[True, False], default=param[1]))

conditionals = [
    InCondition(child=Categorical(name="sat.asymm_branch.all",items=[True, False], default=False),
                parent=cs["sat.asymm_branch"],
                values=[True]),
    InCondition(child=Categorical(name="sat.asymm_branch.sampled", items=[True, False], default=True),
                parent=cs["sat.asymm_branch"],
                values=[True]),
    InCondition(child=Categorical(name="sat.asymm_branch.delay", items=int_categories, default=1),
                parent=cs["sat.asymm_branch"],
                values=[True]),
    InCondition(child=Categorical(name="sat.asymm_branch.limit", items=int_categories, default=100000000),
                parent=cs["sat.asymm_branch"],
                values=[True]),
    InCondition(child=Categorical(name="sat.asymm_branch.rounds", items=int_categories, default=2),
                parent=cs["sat.asymm_branch"],
                values=[True]),
]

for param in conditionals:
    cs.add_condition(param)

class CustomCallback(Callback):
    global integer_hyperparameters
    global boolean_hyperparameters
    global int_categories

    def _init_int(self, param):
        self.categories[param] = {}
        for ic in int_categories:
            self.categories[param][ic] = []

    def _init_bool(self, param):
        self.categories[param] = { 0 : [], 1 : [] }
    
    def __init__(self) -> None:
        self.categories = {}
        for param in integer_hyperparameters:
            self._init_int(f"sat.{param[0]}")
        for param in boolean_hyperparameters:
            self._init_bool(f"sat.{param[0]}")
        self._init_bool("sat.asymm_branch.all")
        self._init_bool("sat.asymm_branch.sampled")
        self._init_int("sat.asymm_branch.delay")
        self._init_int("sat.asymm_branch.limit")
        self._init_int("sat.asymm_branch.rounds")



    def on_start(self, smbo: smac.main.smbo.SMBO) -> None:
        print("Let's start!")
        print("")

    def get_categories_avg(self):
        res = {}
        for param in self.categories:
            res[param] = {}
            for v in self.categories[param]:
                vs = self.categories[param][v]
                if len(vs) == 0:
                    continue
                res[param][v] = sum(vs) / len(vs)
        return res

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        incumbent = smbo.intensifier.get_incumbent()
        incdict = incumbent.get_dictionary()

        v = smbo.runhistory.get_cost(incumbent)
        print(f"Current incumbent: {incdict}")
        print(f"Current incumbent value: {v}")
        print("")

        for param in incdict:
            print(param)
            param_value = incdict[param]
            if param_value == True:
                self.categories[param][1].append(v)
            elif param_value == False:
                self.categories[param][0].append(v)
            else:
                for upper in int_categories:
                    if upper >= int(param_value):
                        self.categories[param][upper].append(v)
                        break

        # print(self.get_categories_avg())
        # print("")

        return None

print(cs)

scenario = Scenario(cs, deterministic=True, n_trials=1000, use_default_config=True)

custom_callback = CustomCallback()

initial_design = AlgorithmConfigurationFacade.get_initial_design(
    scenario,
    additional_configs=[cs.get_default_configuration()],
)

smac = AlgorithmConfigurationFacade(scenario, train, initial_design=initial_design, overwrite=False, callbacks=[custom_callback])
print(smac.runhistory.finished)
print(smac.optimize())
print(custom_callback.get_categories_avg())
# print(categories)

# for k, v in categories.items():
#     vv = []
#     for i in v:
#         if len(v[i]) != 0:
#             vv.append([i, sum(v[i]) / len(v[i])])
#     print(k, vv, sep=" ")

print(res)
