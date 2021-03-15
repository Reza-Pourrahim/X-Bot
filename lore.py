import numpy as np

from functools import partial

from scipy.spatial.distance import cdist

from util import record2str

from explanation import Explanation
from decision_tree import learn_local_decision_tree
from neighborgen_lore import LoreNeighborhoodGenerator
from rule import get_rule, get_counterfactual_rules
from util import neuclidean


def default_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


# LOcal Rule-based Explanation Method
class LORE(object):

    def __init__(self, K, blackbox, feature_names, class_name, class_values, numeric_columns, features_map,
                 categorical_columns, neigh_type='ngmusx', samples=1000, filter_crules=True, kernel_width=None, kernel=None,
                 random_state=None, verbose=False, **kwargs):

        self.random_state = random_state
        self.K = K
        self.blackbox = blackbox
        self.feature_names = feature_names
        self.class_name = class_name
        self.class_values = class_values
        self.numeric_columns = numeric_columns
        self.features_map = features_map
        self.categorical_columns = categorical_columns
        self.neigh_type = neigh_type
        self.samples = samples
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

    def explain_instance(self, x, samples=1000, use_weights=True, metric=neuclidean, nbr_runs=10, exemplar_num=5):

        if isinstance(samples, int):
            if self.verbose:
                print('generating neighborhood - %s' % self.neigh_type)
            Z = self.neighgen_fn(x, samples, nbr_runs, categorical_columns=self.categorical_columns)
        else:
            Z = samples

        Yb = self.blackbox.predict(Z)

        if self.verbose:
            neigh_class, neigh_counts = np.unique(Yb, return_counts=True)
            neigh_class_counts = {self.class_values[k]: v for k, v in zip(neigh_class, neigh_counts)}

            print('synthetic neighborhood class counts %s' % neigh_class_counts)

        weights = None if not use_weights else self.__calculate_weights__(Z, metric)

        # binary, multiclass, multilabel all together
        exp = self.__explain_tabular_instance_single_tree(x, Z, Yb, weights, exemplar_num=exemplar_num)

        return exp

    def __calculate_weights__(self, Z, metric):
        if np.max(Z) != 1 and np.min(Z) != 0:
            Zn = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
            distances = cdist(Zn, Zn[0].reshape(1, -1), metric=metric).ravel()
        else:
            distances = cdist(Z, Z[0].reshape(1, -1), metric=metric).ravel()
        weights = self.kernel(distances)
        return weights

    def __explain_tabular_instance_single_tree(self, x, Z, Yb, weights, exemplar_num=5):

        if self.verbose:
            print('learning local decision tree')

        idx_train = len(Z) - int(len(Z) * 0.05)
        dt = learn_local_decision_tree(Z[:idx_train], Yb[:idx_train], weights[:idx_train],
                                       self.class_values, prune_tree=False)
        Yc = dt.predict(Z)

        # Return the mean accuracy on the given test data and labels.
        fidelity = dt.score(Z, Yb, sample_weight=weights)

        if self.verbose:
            print('retrieving explanation')

        # Factual and Counter-factual Rule
        rule = get_rule(x, dt, self.feature_names, self.class_name, self.class_values, self.numeric_columns)
        crules, deltas = get_counterfactual_rules(x, Yc[0], dt, Z, Yc, self.feature_names, self.class_name,
                                                  self.class_values, self.numeric_columns, self.features_map,
                                                  self.features_map_inv, self.filter_crules)
        # Feature Importance
        feature_importance = self.get_feature_importance(dt, x)

        # Exemplar and Counter-exemplar
        exemplars_rec, cexemplars_rec = self.get_exemplars_cexemplars(dt, x, exemplar_num)
        exemplars = self.get_exemplars_str(exemplars_rec)

        cexemplars = self.get_exemplars_str(cexemplars_rec)

        # Explanation
        exp = Explanation()
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.rule = rule
        exp.crules = crules
        exp.deltas = deltas
        exp.dt = dt
        exp.fidelity = fidelity
        exp.feature_importance = feature_importance
        exp.exemplars = exemplars
        exp.cexemplars = cexemplars

        return exp

    def get_exemplars_str(self, exemplars_rec):
        exemplars = '\n'.join([record2str(s, self.feature_names, self.numeric_columns) for s in exemplars_rec])
        return exemplars

    def get_exemplars_cexemplars(self, dt, x, n):
        leave_id_x = dt.apply(x.reshape(1, -1))
        leave_id_K = dt.apply(self.K)

        exemplar_idx = np.where(leave_id_K == leave_id_x)
        exemplar_vals = self.K[exemplar_idx]

        cexemplar_idx = np.where(leave_id_K != leave_id_x)
        cexemplar_vals = self.K[cexemplar_idx]

        # find instance x in obtained list and remove it
        idx_to_remove = np.where((exemplar_vals == x).all(axis=1))[0]
        if idx_to_remove.any():
            exemplar_vals = np.delete(exemplar_vals, idx_to_remove, axis=0)

        distance_x_exemplar = cdist(x.reshape(1, -1), exemplar_vals, metric='euclidean').ravel()
        distance_x_cexemplar = cdist(x.reshape(1, -1), cexemplar_vals, metric='euclidean').ravel()

        if len(exemplar_vals) < n or len(cexemplar_vals) < n:
            print('maximum number of exemplars and counter-exemplars founded is : %s, %s', len(exemplar_vals),
                  len(cexemplar_vals))
        else:
            first_n_dist_id = distance_x_exemplar.argsort()[:n]
            first_n_exemplar = exemplar_vals[first_n_dist_id]

            first_n_dist_id_c = distance_x_cexemplar.argsort()[:n]
            first_n_cexemplar = cexemplar_vals[first_n_dist_id_c]

        return first_n_exemplar, first_n_cexemplar

    def get_feature_importance(self, dt, x):
        att_list = []
        # dt.apply: Return the index of the leaf that each sample is predicted as.
        leave_id_dt = dt.apply(x.reshape(1, -1))
        node_index_dt = dt.decision_path(x.reshape(1, -1)).indices
        feature_dt = dt.tree_.feature
        for node_id in node_index_dt:
            if leave_id_dt[0] == node_id:
                break
            else:
                att = self.feature_names[feature_dt[node_id]]
                att_list.append(att)
        # Feature importance
        feature_importance_all = dt.feature_importances_
        dict_feature_importance = dict(zip(self.feature_names, feature_importance_all))
        feature_importance = {k: v for k, v in dict_feature_importance.items() if k in att_list}
        return feature_importance
