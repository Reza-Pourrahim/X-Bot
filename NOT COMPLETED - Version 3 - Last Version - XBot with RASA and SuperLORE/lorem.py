import numpy as np
from joblib import Parallel, delayed
import itertools
import multiprocessing as ml
import math
import pickle
from functools import partial
from surrogate import *
from scipy.spatial.distance import cdist
from decision_tree import learn_local_decision_tree

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from rule import Rule, compact_premises, get_counterfactual_rules_supert, get_rule_supert

from explanation import Explanation, MultilabelExplanation
from neighgen import RandomGenerator, GeneticGenerator, RandomGeneticGenerator, ClosestInstancesGenerator, CFSGenerator
from neighgen import GeneticProbaGenerator, RandomGeneticProbaGenerator
from rule import get_rule, get_counterfactual_rules
from util import calculate_feature_values, neuclidean, multilabel2str, multi_dt_predict, record2str
from discretizer import *
from encdec import *


def default_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


# LOcal Rule-based Explanation Method
class LOREM(object):

    def __init__(self, K, bb_predict, feature_names, class_name, class_values, numeric_columns, features_map,
                 neigh_type='genetic', categorical_use_prob=True, continuous_fun_estimation=False,
                 size=1000, ocr=0.1, multi_label=False, one_vs_rest=False, filter_crules=True, init_ngb_fn=True,
                 kernel_width=None, kernel=None, random_state=None, encdec = None, dataset = None, binary=False, discretize=True, verbose=False, extreme_fidelity = False, **kwargs):

        self.random_state = random_state
        self.bb_predict = bb_predict
        self.class_name = class_name
        self.feature_names = feature_names
        self.class_values = class_values
        self.numeric_columns = numeric_columns
        self.features_map = features_map
        self.neigh_type = neigh_type
        self.multi_label = multi_label
        self.one_vs_rest = one_vs_rest
        self.filter_crules = self.bb_predict if filter_crules else None
        self.binary = binary
        self.verbose = verbose
        self.discretize = discretize
        self.extreme_fidelity = extreme_fidelity
        if encdec is not None:
            self.dataset = dataset
            if encdec == 'target':
                print('self. dataset ', self.dataset.columns)
                self.encdec = MyTargetEnc(self.dataset, self.class_name)
                self.encdec.enc_fit_transform()
            elif encdec == 'onehot':
                self.encdec = OneHotEnc(self.dataset, self.class_name)
                self.encdec.enc_fit_transform()
            Y = self.bb_predict(K)
            self.K = self.encdec.enc(K, Y)
        else:
            self.encdec = None
            self.K = K
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

        if init_ngb_fn:
            self.__init_neighbor_fn(ocr, categorical_use_prob, continuous_fun_estimation, size, kwargs)

    def explain_instance(self, x, samples=1000, use_weights=True, metric=neuclidean):

        if isinstance(samples, int):
            if self.verbose:
                print('generating neighborhood - %s' % self.neigh_type)
            Z = self.neighgen_fn(x, samples)
        else:
            Z = samples
        print('z shape ', Z.shape)
        Yb = self.bb_predict(Z)
        print('unique values ', np.unique(Yb))
        if self.multi_label:
            Z = np.array([z for z, y in zip(Z, Yb) if np.sum(y) > 0])
            Yb = self.bb_predict(Z)

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
        dt = learn_local_decision_tree(Z[:idx_train], Yb[:idx_train], weights[:idx_train], self.class_values, self.multi_label, self.one_vs_rest,
                                       prune_tree=False)
        Yc = dt.predict(Z)

        fidelity = dt.score(Z, Yb, sample_weight=weights)

        if self.verbose:
            print('retrieving explanation')

        rule = get_rule(x, self.bb_predict(x.reshape(1,-1)), dt, self.feature_names, self.class_name, self.class_values, self.numeric_columns,
                        self.multi_label)
        crules, deltas = get_counterfactual_rules(x, Yc[0], dt, Z, Yc, self.feature_names, self.class_name,
                                                  self.class_values, self.numeric_columns, self.features_map,
                                                  self.features_map_inv, bb_predict=self.bb_predict, multi_label=self.multi_label)

        exp = Explanation()
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.rule = rule
        exp.crules = crules
        exp.deltas = deltas
        exp.dt = dt
        exp.fidelity = fidelity

        return exp

    def multi_neighgen_fn_parallel(self, x, runs, samples, n_jobs = 2 ):
        Z_list = [list() for i in range(runs)]
        Z_list = Parallel(n_jobs=n_jobs, verbose=self.verbose, prefer='threads')(
            delayed(self.neighgen_fn)(x, samples)
            for i in range(runs))
        return Z_list

    def multi_neighgen_fn(self, x, runs, samples, kwargs=None):
        Z_list = list()
        for i in range(runs):
            if self.verbose:
                print('generating neighborhood [%s/%s] - %s' % (i, runs, self.neigh_type))
                #print(samples, x)
            Z = self.neighgen_fn(x, samples)
            Z_list.append(Z)
        return Z_list

    def get_feature_importance_supert(self, dt, x, tot_samples):
        dt.set_impurity(dt)
        dt.calculate_features_importance(tot_samples)
        all_features = dt.calculate_all_importances()
        single_features = dt.calculate_fi_path(x)
        return single_features, all_features

    def get_feature_importance_binary(self, dt, x):
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
        feature_importance_rule = {k: v for k, v in dict_feature_importance.items() if k in att_list}
        return feature_importance_rule, dict_feature_importance

    '''Explain Instance Stable
            x the instance to explain
            samples the number of samples to generate during the neighbourhood generation
            use weights True or False
            metric default is neuclidean, it is the metric employed to measure the distance between records 
            runs number of times the neighbourhood generation is done
            exemplar_num number of examplars to retrieve
            kwargs a dictionary in which add the parameters needed for cfs generation'''

    # qui l'istanza arriva originale
    def explain_instance_stable(self, x, samples=100, use_weights=True, metric=neuclidean, runs=3, exemplar_num=5,
                                n_jobs=-1, kwargs=None):

        if self.multi_label:
            print('Not yet implemented')
            raise Exception

        if self.encdec is not None:
            y = self.bb_predict(x.reshape(1, -1))
            x = self.encdec.enc(x, y)

        if isinstance(samples, int):
            if self.neigh_type == 'cfs':
                Z_list = self.multi_neighgen_fn_parallel(x, runs, samples, n_jobs)
            else:
                Z_list = self.multi_neighgen_fn(x, runs, samples, kwargs)
        else:
            Z_list = samples

        Yb_list = list()
        if self.encdec is not None:
            for Z in Z_list:
                Z = self.encdec.dec(Z)
                Yb = self.bb_predict(Z)
                Yb_list.append(Yb)
        else:
            for Z in Z_list:
                Yb = self.bb_predict(Z)
                Yb_list.append(Yb)

        if self.verbose:
            neigh_class_counts_list = list()
            for Yb in Yb_list:
                neigh_class, neigh_counts = np.unique(Yb, return_counts=True)
                neigh_class_counts = {self.class_values[k]: v for k, v in zip(neigh_class, neigh_counts)}
                neigh_class_counts_list.append(neigh_class_counts)

            for neigh_class_counts in neigh_class_counts_list:
                print('Synthetic neighborhood class counts %s' % neigh_class_counts)

        weights_list = list()
        for Z in Z_list:
            weights = None if not use_weights else self.__calculate_weights__(Z, metric)
            weights_list.append(weights)

        if self.verbose:
            print('Learning local decision trees')

        # discretize the data employed for learning decision tree
        if self.discretize:
            Z = np.concatenate(Z_list)
            Yb = np.concatenate(Yb_list)

            discr = RMEPDiscretizer()
            discr.fit(Z, Yb)
            temp = list()
            for Zl in Z_list:
                temp.append(discr.transform(Zl))
            Z_list = temp

        dt_list = [DecTree() for i in range(runs)]
        dt_list = Parallel(n_jobs=n_jobs, verbose=self.verbose,prefer='threads')(
            delayed(t.learn_local_decision_tree)(Zl, Yb, weights, self.class_values)
            for Zl, Yb, weights, t in zip(Z_list, Yb_list, weights_list, dt_list))

        Z = np.concatenate(Z_list)
        Yb = np.concatenate(Yb_list)
        if self.verbose:
            print('Pruning decision trees')
        surr = SuperTree()
        for t in dt_list:
            surr.prune_duplicate_leaves(t)
        if self.verbose:
            print('Merging decision trees')

        weights_list = list()
        for Zl in Z_list:
            weights = None if not use_weights else self.__calculate_weights__(Zl, metric)
            weights_list.append(weights)
        weights = np.concatenate(weights_list)
        n_features = list()
        for d in dt_list:
            n_features.append(list(range(0, len(self.feature_names))))
        roots = np.array([surr.rec_buildTree(t, FI_used) for t, FI_used in zip(dt_list, n_features)])

        superT = surr.mergeDecisionTrees(roots, num_classes=np.unique(Yb).shape[0], verbose=False)

        if self.binary:
            superT = surr.supert2b(superT, Z)
            Yb = superT.predict(Z)
            fidelity = superT.score(Z, Yb, sample_weight=weights)
        else:
            Yz = superT.predict(Z)
            fidelity = accuracy_score(Yb, Yz)

        if self.extreme_fidelity:
            res = superT.predict(x)
            if res != y:
                raise Exception('The prediction of the surrogate model is differen wrt the black box')

        Yc = superT.predict(X=Z)

        if self.verbose:
            print('Retrieving explanation')
        x = x.flatten()
        if self.binary:
            rule = get_rule(x, self.bb_predict(x.reshape(1, -1)), superT, self.feature_names, self.class_name, self.class_values,
                            self.numeric_columns, encdec=self.encdec,
                            multi_label=self.multi_label)
        else:
            rule = get_rule_supert(x, superT, self.feature_names, self.class_name, self.class_values,
                                   self.numeric_columns,
                                   self.multi_label, encdec=self.encdec)
        if self.binary:
            crules, deltas = get_counterfactual_rules(x, Yc[0], superT, Z, Yc, self.feature_names,
                                                      self.class_name, self.class_values, self.numeric_columns,
                                                      self.features_map, self.features_map_inv, encdec=self.encdec,
                                                      filter_crules = self.filter_crules)
        else:
            crules, deltas = get_counterfactual_rules_supert(x, Yc[0], superT, Z, Yc, self.feature_names,
                                                             self.class_name, self.class_values, self.numeric_columns,
                                                             self.features_map, self.features_map_inv,
                                                             filter_crules = self.filter_crules)

        exp = Explanation()
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.rule = rule
        exp.crules = crules
        exp.deltas = deltas
        exp.dt = superT
        exp.fidelity = fidelity
        # Feature Importance
        if self.binary:
            feature_importance, feature_importance_all = self.get_feature_importance_binary(superT, x)

        else:
            feature_importance, feature_importance_all = self.get_feature_importance_supert(superT, x, len(Yb))
        # Exemplar and Counter-exemplar
        exemplars_rec, cexemplars_rec = self.get_exemplars_cexemplars_binary(superT, x, exemplar_num)
        exemplars = self.get_exemplars_str(exemplars_rec)
        cexemplars = self.get_exemplars_str(cexemplars_rec)
        exp.feature_importance = feature_importance
        exp.feature_importance_all = feature_importance_all
        exp.exemplars = exemplars
        exp.cexemplars = cexemplars
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
                dt = learn_local_decision_tree(Z[:idx_train], Yb[:idx_train, l], weights[:idx_train], self.class_values, self.multi_label,
                                               self.one_vs_rest, prune_tree=False)
                Yc = dt.predict(Z)
                class_values = [0, 1]
                rule = get_rule(x, self.bb_predict(x.reshape(1,-1)), dt, self.feature_names, self.class_name[l], class_values, self.numeric_columns,
                                multi_label=False)
                crules, deltas = get_counterfactual_rules(x, self.bb_predict(x.reshape(1,-1)), dt, Z, Yc, self.feature_names,
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

    def __init_neighbor_fn(self, ocr, categorical_use_prob, continuous_fun_estimation, size, kwargs):

        neighgen = None
        numeric_columns_index = [i for i, c in enumerate(self.feature_names) if c in self.numeric_columns]

        self.feature_values = None
        if self.neigh_type in ['random', 'genetic', 'rndgen', 'geneticp', 'rndgenp']:
            if self.verbose:
                print('calculating feature values')
            self.feature_values = calculate_feature_values(self.K, numeric_columns_index,
                                                           categorical_use_prob=categorical_use_prob,
                                                           continuous_fun_estimation=continuous_fun_estimation,
                                                           size=size)

        nbr_features = len(self.feature_names)
        nbr_real_features = self.K.shape[1]

        if self.neigh_type in ['genetic', 'rndgen', 'geneticp', 'rndgenp']:
            alpha1 = kwargs.get('alpha1', 0.5)
            alpha2 = kwargs.get('alpha2', 0.5)
            metric = kwargs.get('metric', neuclidean)
            ngen = kwargs.get('ngen', 10)
            mutpb = kwargs.get('mutpb', 0.5)
            cxpb = kwargs.get('cxpb', 0.7)
            tournsize = kwargs.get('tournsize', 3)
            halloffame_ratio = kwargs.get('halloffame_ratio', 0.1)
            random_seed = self.random_state

            if self.neigh_type == 'genetic':
                neighgen = GeneticGenerator(self.bb_predict, self.feature_values, self.features_map, nbr_features,
                                            nbr_real_features, numeric_columns_index, ocr=ocr, alpha1=alpha1,
                                            alpha2=alpha2, metric=metric, ngen=ngen,
                                            mutpb=mutpb, cxpb=cxpb, tournsize=tournsize,
                                            halloffame_ratio=halloffame_ratio, random_seed=random_seed, encdec=self.encdec,
                                            verbose=self.verbose)
            elif self.neigh_type == 'rndgen':
                neighgen = RandomGeneticGenerator(self.bb_predict, self.feature_values, self.features_map,
                                                  nbr_features, nbr_real_features, numeric_columns_index,
                                                  ocr=ocr, alpha1=alpha1, alpha2=alpha2,
                                                  metric=metric, ngen=ngen, mutpb=mutpb, cxpb=cxpb,
                                                  tournsize=tournsize, halloffame_ratio=halloffame_ratio,
                                                  random_seed=random_seed, encdec=self.encdec, verbose=self.verbose)
            elif self.neigh_type == 'geneticp':
                bb_predict_proba = kwargs.get('bb_predict_proba', None)
                neighgen = GeneticProbaGenerator(self.bb_predict, self.feature_values, self.features_map, nbr_features,
                                                 nbr_real_features, numeric_columns_index, ocr=ocr, alpha1=alpha1,
                                                 alpha2=alpha2, metric=metric, ngen=ngen,
                                                 mutpb=mutpb, cxpb=cxpb, tournsize=tournsize,
                                                 halloffame_ratio=halloffame_ratio,
                                                 bb_predict_proba=bb_predict_proba,
                                                 random_seed=random_seed, encdec=self.encdec,
                                                 verbose=self.verbose)

            elif self.neigh_type == 'rndgenp':
                bb_predict_proba = kwargs.get('bb_predict_proba', None)
                neighgen = RandomGeneticProbaGenerator(self.bb_predict, self.feature_values, self.features_map,
                                                       nbr_features, nbr_real_features, numeric_columns_index,
                                                       ocr=ocr, alpha1=alpha1, alpha2=alpha2,
                                                       metric=metric, ngen=ngen, mutpb=mutpb, cxpb=cxpb,
                                                       tournsize=tournsize, halloffame_ratio=halloffame_ratio,
                                                       bb_predict_proba=bb_predict_proba,
                                                       random_seed=random_seed, encdec=self.encdec, verbose=self.verbose)

        elif self.neigh_type == 'random':
            neighgen = RandomGenerator(self.bb_predict, self.feature_values, self.features_map, nbr_features,
                                       nbr_real_features, numeric_columns_index, ocr=ocr)
        elif self.neigh_type == 'closest':
            Kc = kwargs.get('Kc', None)
            k = kwargs.get('k', None)
            type = kwargs.get('core_neigh_type', 'simple')
            alphaf = kwargs.get('alphaf', 0.5)
            alphal = kwargs.get('alphal', 0.5)
            metric_features = kwargs.get('metric_features', neuclidean)
            metric_labels = kwargs.get('metric_labels', neuclidean)
            neighgen = ClosestInstancesGenerator(self.bb_predict, self.feature_values, self.features_map, nbr_features,
                                                 nbr_real_features, numeric_columns_index, ocr=ocr,
                                                 K=Kc, rK=self.K, k=k, core_neigh_type=type, alphaf=alphaf,
                                                 alphal=alphal, metric_features=metric_features,
                                                 metric_labels=metric_labels, categorical_use_prob=categorical_use_prob,
                                                 continuous_fun_estimation=continuous_fun_estimation, size=size, encdec=self.encdec,
                                                 verbose=self.verbose)
        elif self.neigh_type == 'cfs':
            if self.verbose:
                print('Neigh kind ', self.neigh_type)
                print('sampling kind ', kwargs.get('kind', None))
            neighgen = CFSGenerator(self.bb_predict, self.feature_values, self.features_map, nbr_features,
                                                 nbr_real_features, numeric_columns_index,
                                    ocr=ocr,
                                    kind=kwargs.get('kind', None),
                                    sampling_kind=kwargs.get('sampling_kind', None),
                                    #vicinity_sampler_kwargs=kwargs.get('vicinity_sampler_kwargs', None),
                                    stopping_ratio=kwargs.get('stopping_ratio', 0.01),
                                    n_batch=kwargs.get('n_batch', 560),
                                    check_upper_threshold=kwargs.get('check_upper_threshold', True),
                                    final_counterfactual_search=kwargs.get('final_counterfactual_search',True),
                                    verbose=kwargs.get('verbose', False),
                                    custom_sampling_threshold=kwargs.get('custom_sampling_threshold', None),
                                    custom_closest_counterfactual=kwargs.get('custom_closest_counterfactual',None),
                                    n=kwargs.get('n', 10000), balance=kwargs.get('balance', None),
                                    forced_balance_ratio = kwargs.get('forced_balance_ratio', 0.5),
                                    cut_radius=kwargs.get('cut_radius', False),
                                    downward_only=kwargs.get('downward_only', None),
                                    encdec=self.encdec
                                    )
        else:
            print('unknown neighborhood generator')
            raise Exception

        self.neighgen_fn = neighgen.generate

    def get_exemplars_str(self, exemplars_rec):
        exemplars = '\n'.join([record2str(s, self.feature_names, self.numeric_columns, encdec=self.encdec) for s in exemplars_rec])
        return exemplars

    def get_exemplars_cexemplars_binary(self, dt, x, n):
        if self.encdec is not None:
            dataset = self.dataset
            labels = dataset.pop(self.class_name)
            dataset = self.encdec.enc(dataset.values, labels.values)
            leave_id_K = dt.apply(dataset)
        else:
            leave_id_K = dt.apply(self.K)

        leave_id_x = dt.apply(x.reshape(1, -1))
        exemplar_idx = np.where(leave_id_K == leave_id_x)
        if self.encdec is not None:
            exemplar_vals = dataset[exemplar_idx]
        else:
            exemplar_vals = self.K[exemplar_idx]

        cexemplar_idx = np.where(leave_id_K != leave_id_x)
        if self.encdec is not None:
            cexemplar_vals = dataset[cexemplar_idx]
        else:
            cexemplar_vals = self.K[cexemplar_idx]

        # find instance x in obtained list and remove it
        idx_to_remove = None
        if x in exemplar_vals:
            idx_to_remove = np.where((exemplar_vals == x).all(axis=1))[0]
        if idx_to_remove is not None:
            exemplar_vals = np.delete(exemplar_vals, idx_to_remove, axis=0)

        distance_x_exemplar = cdist(x.reshape(1, -1), exemplar_vals, metric='euclidean').ravel()
        distance_x_cexemplar = cdist(x.reshape(1, -1), cexemplar_vals, metric='euclidean').ravel()

        if len(exemplar_vals) < n or len(cexemplar_vals) < n:
            if self.verbose:
                print('maximum number of exemplars and counter-exemplars founded is : %s, %s', len(exemplar_vals),
                  len(cexemplar_vals))
            n = min(len(cexemplar_vals),len(exemplar_vals))
        first_n_dist_id = distance_x_exemplar.argsort()[:n]
        first_n_exemplar = exemplar_vals[first_n_dist_id]

        first_n_dist_id_c = distance_x_cexemplar.argsort()[:n]
        first_n_cexemplar = cexemplar_vals[first_n_dist_id_c]

        return first_n_exemplar, first_n_cexemplar


    def get_exemplars_cexemplars_supert(self, dt, x, n):
        if self.encdec is not None:
            dataset = self.dataset
            labels = dataset.pop(self.class_name)
            dataset = self.encdec.enc(dataset.values, labels.values)
            leave_id_K = dt.apply(dataset)
        else:
            leave_id_K = dt.apply(self.K)

        print('leave id applied ', leave_id_K)

        leave_id_x = dt.apply(x.reshape(1, -1))


        exemplar_idx = np.where(leave_id_K == leave_id_x)
        print('exemplar idx ', len(exemplar_idx))

        if self.encdec is not None:
            exemplar_vals = dataset[exemplar_idx]
        else:
            exemplar_vals = self.K[exemplar_idx]
        cexemplar_idx = np.where(leave_id_K != leave_id_x)
        print('cexemplar idx ', len(cexemplar_idx))
        if self.encdec is not None:
            cexemplar_vals = dataset[cexemplar_idx]
        else:
            cexemplar_vals = self.K[cexemplar_idx]
        print('exemplar and counter exemplars ', exemplar_vals, cexemplar_vals)
        # find instance x in obtained list and remove it
        idx_to_remove = None
        if x in exemplar_vals:
            print('cerco la x')
            idx_to_remove = np.where((exemplar_vals == x).all(axis=1))[0]
        if idx_to_remove is not None:
            print('la tolgo')
            exemplar_vals = np.delete(exemplar_vals, idx_to_remove, axis=0)

        distance_x_exemplar = cdist(x.reshape(1, -1), exemplar_vals, metric='euclidean').ravel()
        distance_x_cexemplar = cdist(x.reshape(1, -1), cexemplar_vals, metric='euclidean').ravel()

        if len(exemplar_vals) < n or len(cexemplar_vals) < n:
            if self.verbose:
                print('maximum number of exemplars and counter-exemplars founded is : %s, %s', len(exemplar_vals),
                  len(cexemplar_vals))
            n = min(len(cexemplar_vals),len(exemplar_vals))
        first_n_dist_id = distance_x_exemplar.argsort()[:n]
        first_n_exemplar = exemplar_vals[first_n_dist_id]

        first_n_dist_id_c = distance_x_cexemplar.argsort()[:n]
        first_n_cexemplar = cexemplar_vals[first_n_dist_id_c]

        return first_n_exemplar, first_n_cexemplar

    def explain_set_instances_stable(self, X, n_workers, title, runs=3, n_jobs =4, n_samples=1000, exemplar_num=5, use_weights=True, metric=neuclidean, kwargs=None):
        # for parallelization
        items_for_worker = math.ceil( len(X)/ float(n_workers))
        start = 0
        print(items_for_worker)
        end = int(items_for_worker)
        processes = list()
        # create workers
        print("Dispatching jobs to workers...\n")
        for i in range(0, n_workers):
            dataset = X[start:end]
            process = ml.Process(target=self.explain_workers_stable, args=(i, dataset, title, n_samples, use_weights, metric, runs, n_jobs, exemplar_num, kwargs))
            processes.append(process)
            process.start()

            if end > (len(X)-1):

                workers = n_workers - 1
                break
            start = end
            end += int(items_for_worker)

        # join workers
        for i in range(0, workers):
            processes[i].join()
        print("All workers joint.\n")

    def explain_workers_stable(self, i, dataset, title, n_samples, use_wieghts, metric, runs=3, n_jobs =4, exemplar_num=5, kwargs=None):
        results = list()
        count = 0
        for d in dataset:
            print(i, count)
            count += 1
            d = np.array(d)
            try:
                exp = self.explain_instance_stable(d, samples=n_samples, use_weights=use_wieghts, metric=metric, runs=runs, exemplar_num=exemplar_num, n_jobs=n_jobs, kwargs=kwargs)
                results.append((d, exp))
            except:
                print('counterfactual not found')
                continue
        title = 'explanations_lore' + title + '_' + str(i) + '.p'
        with open(title, "ab") as pickle_file:
            pickle.dump(results, pickle_file)
        return
