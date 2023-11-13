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

file = open("benches_alia_20_sec_train", "r")
benchmarks = list(file.readlines())
file.close()

res = []

print(len(benchmarks))
def train(config: ConfigurationSpace, seed):
    global benchmarks
    global res

    launches = 1
    r = 0

    for i in range(launches):
        times = []
        for lines in benchmarks:
            s = Solver()
            s.set("timeout", 600000)
            set_param(**config)
            set_param("sat.random_seed", seed)
            set_param("nlsat.seed", seed)
            set_param("fp.spacer.random_seed", seed)
            set_param("smt.random_seed", seed)
            set_param("sls.random_seed", seed)

            benchmark = lines.split(", ")[0]    
            exps = parse_smt2_file(benchmark)
            start = time.time()
            s.assert_exprs(exps)
            s.check()
            times.append(time.time() - start)
        r += sum(times)
    r /= launches
    res.append(r)

    return r

cs = ConfigurationSpace()

boolean_hyperparameters = {
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

        return None

print(cs)

scenario = Scenario(cs, deterministic=True, n_trials=3000, use_default_config=True)

custom_callback = CustomCallback()

initial_design = AlgorithmConfigurationFacade.get_initial_design(
    scenario,
    additional_configs=[cs.get_default_configuration()],
)

smac = AlgorithmConfigurationFacade(scenario, train, initial_design=initial_design, overwrite=False, callbacks=[custom_callback])
print(smac.runhistory.finished)
print(smac.optimize())
print(custom_callback.get_categories_avg())
print(res)
