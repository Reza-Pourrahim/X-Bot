import numpy as np

import itertools
from functools import partial

from scipy.spatial.distance import cdist

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from rule import Rule, compact_premises

from explanation import Explanation, MultilabelExplanation
from decision_tree import learn_local_decision_tree
from neighborgen_lore import LoreNeighborhoodGenerator
from rule import get_rule, get_counterfactual_rules
from util import calculate_feature_values, neuclidean, multilabel2str, multi_dt_predict


def default_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


# LOcal Rule-based Explanation Method
class LORE(object):

    def __init__(self, K, blackbox, feature_names, class_name, class_values, numeric_columns, features_map,
                 neigh_type='ngmusx', samples=1000, multi_label=False, one_vs_rest=False, filter_crules=True,
                 kernel_width=None, kernel=None, random_state=None, verbose=False, **kwargs):

        self.random_state = random_state
        self.K = K
        self.blackbox = blackbox
        self.feature_names = feature_names
        self.class_name = class_name
        self.class_values = class_values
        self.numeric_columns = numeric_columns
        self.features_map = features_map
        self.neigh_type = neigh_type
        self.samples = samples
        self.multi_label = multi_label
        self.one_vs_rest = one_vs_rest
        self.filter_crules = self.blackbox.predict if filter_crules else None
        self.verbose = verbose

        self.features_map_inv = None
        if self.features_map:
            self.features_map_inv = dict()
            for idx, idx_dict in self.features_map.items():
                for k, v in idx_dict.items():
                    self.features_map_inv[v] = idx

        kernel_width = np.sqrt(len(self.feature_names)) * .75 if kernel_width is None else kernel_width
        self.kernel_width = float(kernel_width)

        kernel = default_kernel if kernel is None else kernel
        self.kernel = partial(kernel, kernel_width=kernel_width)

        np.random.seed(self.random_state)

        self.__init_neighborhood__()

    def __init_neighborhood__(self):
        neighgen_gn = LoreNeighborhoodGenerator(self.blackbox, self.neigh_type, self.samples, self.verbose)
        self.neighgen_fn = neighgen_gn.generate_fn



    def explain_instance(self, x, y_val, blackbox, samples=1000, use_weights=True, metric=neuclidean,nbr_runs=10,
                         verbose=False):

        if isinstance(samples, int):
            if self.verbose:
                print('generating neighborhood - %s' % self.neigh_type)
            Z = self.neighgen_fn(x, samples, nbr_runs)
        else:
            Z = samples

        Yb = self.blackbox.predict(Z)
        if self.multi_label:
            Z = np.array([z for z, y in zip(Z, Yb) if np.sum(y) > 0])
            Yb = self.blackbox.predict(Z)

        if self.verbose:
            if not self.multi_label:
                neigh_class, neigh_counts = np.unique(Yb, return_counts=True)
                neigh_class_counts = {self.class_values[k]: v for k, v in zip(neigh_class, neigh_counts)}
            else:
                neigh_counts = np.sum(Yb, axis=0)
                neigh_class_counts = {self.class_values[k]: v for k, v in enumerate(neigh_counts)}

            print('synthetic neighborhood class counts %s' % neigh_class_counts)

        weights = None if not use_weights else self.__calculate_weights__(Z, metric)

        if self.one_vs_rest and self.multi_label:
            exp = self.__explain_tabular_instance_multiple_tree(x, Z, Yb, weights)
        else:  # binary, multiclass, multilabel all together
            exp = self.__explain_tabular_instance_single_tree(x, Z, Yb, weights)

        return exp

    def __calculate_weights__(self, Z, metric):
        if np.max(Z) != 1 and np.min(Z) != 0:
            Zn = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
            distances = cdist(Zn, Zn[0].reshape(1, -1), metric=metric).ravel()
        else:
            distances = cdist(Z, Z[0].reshape(1, -1), metric=metric).ravel()
        weights = self.kernel(distances)
        return weights

    def __explain_tabular_instance_single_tree(self, x, Z, Yb, weights):

        if self.verbose:
            print('learning local decision tree')

        idx_train = len(Z) - int(len(Z) * 0.05)
        dt = learn_local_decision_tree(Z[:idx_train], Yb[:idx_train], weights[:idx_train],
                                       self.class_values, self.multi_label, self.one_vs_rest,
                                       prune_tree=False)
        Yc = dt.predict(Z)

        fidelity = dt.score(Z, Yb, sample_weight=weights)

        if self.verbose:
            print('retrieving explanation')

        rule = get_rule(x, dt, self.feature_names, self.class_name, self.class_values, self.numeric_columns,
                        self.multi_label)
        crules, deltas = get_counterfactual_rules(x, Yc[0], dt, Z, Yc, self.feature_names, self.class_name,
                                                  self.class_values, self.numeric_columns, self.features_map,
                                                  self.features_map_inv, self.filter_crules, self.multi_label)

        exp = Explanation()
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.rule = rule
        exp.crules = crules
        exp.deltas = deltas
        exp.dt = dt
        exp.fidelity = fidelity

        return exp

    def __explain_tabular_instance_multiple_tree(self, x, Z, Yb, weights):

        dt_list = list()
        premises = list()
        rule_list = list()
        crules_list = list()
        deltas_list = list()
        nbr_labels = len(self.class_name)

        if self.verbose:
            print('learning %s local decision trees' % nbr_labels)

        for l in range(nbr_labels):
            if np.sum(Yb[:, l]) == 0 or np.sum(Yb[:, l]) == len(Yb):
                outcome = 0 if np.sum(Yb[:, l]) == 0 else 1
                rule = Rule([], outcome, [0, 1])
                crules, deltas = list(), list()
                dt = DummyClassifier()
                dt.fit(np.zeros(Z.shape[1]).reshape(1, -1), np.array([outcome]))
            else:
                idx_train = len(Z) - int(len(Z) * 0.05)
                dt = learn_local_decision_tree(Z[:idx_train], Yb[:idx_train, l], weights[:idx_train],
                                               self.class_values, self.multi_label,
                                               self.one_vs_rest, prune_tree=False)
                Yc = dt.predict(Z)
                class_values = [0, 1]
                rule = get_rule(x, dt, self.feature_names, self.class_name[l], class_values, self.numeric_columns,
                                multi_label=False)
                crules, deltas = get_counterfactual_rules(x, Yc[0], dt, Z, Yc, self.feature_names,
                                                          self.class_name[l], class_values, self.numeric_columns,
                                                          self.features_map, self.features_map_inv,
                                                          self.filter_crules, multi_label=False)

            dt_list.append(dt)
            rule_list.append(rule)
            premises.extend(rule.premises)
            crules_list.append(crules)
            deltas_list.append(deltas)

        if self.verbose:
            print('retrieving explanation')

        Yc = multi_dt_predict(Z, dt_list)
        fidelity = accuracy_score(Yb, Yc, sample_weight=weights)

        premises = compact_premises(premises)
        dt_outcome = multi_dt_predict(x.reshape(1, -1), dt_list)[0]
        cons = multilabel2str(dt_outcome, self.class_values)
        rule = Rule(premises, cons, self.class_name)

        exp = MultilabelExplanation()
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.rule = rule
        exp.crules = list(itertools.chain.from_iterable(crules_list))
        exp.deltas = list(itertools.chain.from_iterable(deltas_list))
        exp.dt = dt_list
        exp.fidelity = fidelity

        exp.rule_list = rule_list
        exp.crules_list = crules_list
        exp.deltas_list = deltas_list

        return exp
