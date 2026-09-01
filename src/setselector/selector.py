"""
Selects a representative benchmark set based on runtime and feature data.

Author: Marius Lindauer
Modified by: Tom Schmidt
Date: 1st September 2026
"""

from __future__ import annotations

import math
import operator
import random

from . import clusterKMeans
from .utils.logging import get_logger

log = get_logger("selector")
SEED = 1234

class Selector:
    """
    Selects a representative set which is good for benchmarking.
    """

    def __init__(self, cutoff: int) -> None:
        self.solvers: list[str] = []
        self._feature_data_dic: dict[str, list[float]] = {}
        self._runtime_data_dic: dict[str, list[float]] = {}
        self.runtime_data: list[list[float]] = []
        self.feature_data: list[list[float]] = []
        self._clusters: dict[str, int | str] = {}
        self.samples: list[str] = []
        self.cutoff = cutoff
        self._n_clusters = 10

    def parse_features(self, feature_file: str) -> bool:
        """
        Parse csv feature file.

        :param feature_file: csv file with features (first col instance name)
        :return: True if successful, False otherwise.
        """
        try:
            with open(feature_file, encoding="utf-8") as fh:
                _ = fh.readline()
                for line in fh:
                    parts = line.split(",")
                    inst_name = parts.pop(0)
                    values = []
                    for value in parts:
                        try:
                            # minimal value -1; negative values are missing values (e.g. -512 satzilla)
                            values.append(max(float(value), -1.0))
                        except ValueError:
                            pass
                    if values:  # filter empty lines
                        if inst_name in self._feature_data_dic:
                            log.warning("duplication of feature data for %s, overwriting", inst_name)
                        self._feature_data_dic[inst_name] = values
        except OSError:
            log.error("failed to parse feature file %s", feature_file, exc_info=True)
            return False
        log.debug(">>>Feature Data:<<<")
        log.debug(self._feature_data_dic)
        log.info("Reading Features was successful!")
        return True

    def parse_runtimes(self, runtimefile: str) -> bool:
        """
        Parse csv runtime file.

        :param runtimefile: csv file with runtimes (first col instance name)
        :return: True if successful, False otherwise.
        """
        try:
            with open(runtimefile, encoding="utf-8") as fh:
                header = fh.readline()
                self.solvers = header.replace("\n", "").split(",")
                for line in fh:
                    parts = line.split(",")
                    inst_name = parts.pop(0)
                    values = []
                    for value in parts:
                        try:
                            values.append(min(self.cutoff, float(value)))
                        except ValueError:
                            pass
                    if values:  # filter empty lines
                        if inst_name in self._runtime_data_dic:
                            log.warning("duplication of runtime data for %s, overwriting", inst_name)
                        self._runtime_data_dic[inst_name] = values
        except OSError:
            log.error("failed to parse runtime file %s", runtimefile, exc_info=True)
            return False
        log.debug(">>>Runtime Data:<<<")
        log.debug(self._runtime_data_dic)
        log.info("Reading Runtimes was successful!")
        return True

    def random_test_training_split(self) -> None:
        """
        Split instances in training and test instances; test instances are removed and printed.

        :return: None
        """
        random.seed(SEED)
        instances = list(self._runtime_data_dic.keys())
        n = len(instances)
        log.debug(">>> Test Instances (remaining instances are used as training instances): ")
        for index in range(n // 2):
            s_instance = random.randint(0, n - index - 1)
            instance = instances.pop(s_instance)
            log.debug(instance)
            self._runtime_data_dic.pop(instance)
            self._feature_data_dic.pop(instance, None)

    def join_times_features(self) -> None:
        """
        Join the features and runtimes (intersection of instance names).

        :return: None
        """
        available = 0
        runtime_data_dic_local = {}
        feature_data_dic_local = {}
        length_feats = -1
        for inst, times in self._runtime_data_dic.items():
            if sum(times) == len(times) * self.cutoff:  # filter instances with only timeouts
                self._clusters[inst] = "h"  # mark too hard instances
                continue
            features = self._feature_data_dic.get(inst)
            if length_feats == -1 and features is not None:
                length_feats = len(features)
            if features is None:
                log.warning("there are runtime data but no features available for %s", inst)
                self._clusters[inst] = "f"  # mark instances with failed feature extraction
                continue
            if len(features) != length_feats:
                log.warning("Invalid number of features for %s : %s", inst, len(features))
                self._clusters[inst] = "f"
                continue
            if sum(features) == 0.0:  # error output of feature extraction
                continue
            if math.isnan(sum(features)) or math.isinf(sum(features)):
                log.warning("feature data include NAN or INF in %s", inst)
                self._clusters[inst] = "f"
                continue
            # else everything is ok
            feature_data_dic_local[inst] = features
            runtime_data_dic_local[inst] = times
            self.runtime_data.append(times)
            self.feature_data.append(features)
            available += 1
        self._runtime_data_dic = runtime_data_dic_local
        self._feature_data_dic = feature_data_dic_local
        print(f">> Available Data: {available}")

    def clustering(self, reps: int) -> None:
        """
        Cluster instances (kmeans) based on distance in feature space.

        :param reps: repetitions of clustering (start sensitive)
        :return: None
        """
        # seed, feature, reps, clus, findK, readIn
        cluster_list = clusterKMeans.do_cluster(SEED, self._feature_data_dic, reps, -1, 10, False)
        cluster_index = 0

        for clu in cluster_list:
            for inst in clu.points:
                self._clusters[str(inst)] = cluster_index
            cluster_index += 1
        self._n_clusters = cluster_index + 1

        for inst, cluster in self._clusters.items():
            log.debug("%s,%s", inst, cluster)
        log.info("Data in Clusters: %s", len(self._clusters))

    def select(self, n: int, frac: float, agg: str, dist: str) -> list[str]:
        """
        Select instances based on gaussian or uniform distribution and feature clusters.

        :param n: number of instances to select
        :param frac: maximal fraction of representation of one cluster
        :param agg: how to aggregate instance hardness (ind, avg or min)
        :param dist: select distribution (gauss or uni)
        :return: samples: list of instances
        """
        random.seed(SEED)
        samples: list[str] = []
        sampled = 0
        sorted_inst = self.sort_inst(agg)
        log.info("Number of Clusters %s", self._n_clusters)
        cluster_reps = self._n_clusters * [0.0]
        mean, variance = self.get_runtime_statistics(sorted_inst)
        log.info("Mean: %s\t Variance: %s", mean, variance)
        while sampled < n and sorted_inst != []:
            if dist == "gauss":
                sample = random.gauss(mean, math.sqrt(variance))
            elif dist == "uni":
                sample = random.random() * float(self.cutoff)
            elif dist == "exp":
                sample = random.expovariate(1 / mean)
            elif dist == "log":
                sample = random.lognormvariate(math.log(mean), math.log(math.sqrt(variance)))
            else:
                raise ValueError(f"Unknown distribution: {dist}")
            inst, agg_value = self.find_nearest(sample, sorted_inst)
            inst = inst.split(",")[0]
            log.debug("%s, %s", inst, agg_value)
            cluster = self._clusters[inst]
            log.debug("Cluster: %s", cluster)
            if not self.is_overrepresented(cluster, cluster_reps, frac, n) and samples.count(inst) == 0:
                samples.append(inst)
                sampled += 1
                log.debug("ACCEPTED")
            else:
                log.debug("REJECTED")
        log.info("Remaining Instances: %s", len(sorted_inst))
        self.samples = samples
        return samples

    def remove_too_easy(self, cutoff: float, threshold: float) -> None:
        """
        Remove too easy instances (< threshold*cutoff).

        :param cutoff: of measured runtime
        :param threshold: fraction of cutoff
        :return: None
        """
        removeable = []
        for inst, vec in self._runtime_data_dic.items():
            maxi = max(vec)
            if maxi < cutoff and maxi < threshold * cutoff:
                removeable.append(inst)
                self._clusters[inst] = "e"  # mark too easy instances
        for rem in removeable:
            self._runtime_data_dic.pop(rem)
            self._feature_data_dic.pop(rem)
        log.info("Remaining Instances after Easy Filtering: %s", len(self._runtime_data_dic))

    def sort_inst(self, agg: str) -> list[tuple[str, float]]:
        """
        Sort instances by hardness aggregation.

        :param agg: how to aggregate instance hardness (ind, avg or min)
        :return: sorted_tuples: (instances, instance hardness)
        """
        inst_avg_dic: dict[str, float] = {}
        for inst, vec in self._runtime_data_dic.items():
            if agg == "ind":
                for index, v in enumerate(vec, start=1):
                    inst_avg_dic[f"{inst},{index}"] = v
                continue
            if agg == "avg":
                aggvec = float(sum(vec)) / len(vec)
            elif agg == "min":
                aggvec = float(min(vec))
            else:
                raise ValueError(f"Unknown aggregation: {agg}")
            inst_avg_dic[inst] = aggvec
        return sorted(inst_avg_dic.items(), key=operator.itemgetter(1))

    def find_nearest(self, value: float, sorted_tuples: list[tuple[str, float]]) -> tuple[str, float]:
        """
        Given an instance hardness value, find nearest instance.

        :param value: float
        :param sorted_tuples: (instance, hardness)
        :return: tuple: nearest tuple to value
        """
        last_inst = None
        last_avg = 0.0
        for index, (inst, avg) in enumerate(sorted_tuples):
            if avg > value and last_inst is None:
                sorted_tuples.pop(index)
                return inst, avg
            if avg > value and last_inst is not None:
                if avg - value < last_avg - value:
                    sorted_tuples.pop(index)
                    return inst, avg
                sorted_tuples.pop(index - 1)
                return last_inst, last_avg
            last_avg = avg
            last_inst = inst

        return sorted_tuples.pop(len(sorted_tuples) - 1)

    def get_runtime_statistics(self, sorted_tuples: list[tuple[str, float]]) -> tuple[float, float]:
        """
        Get mean and variance of instance hardnesses.

        :param sorted_tuples: (instance, hardness)
        :return: mean, variance
        """
        total = 0.0
        total_sqr = 0.0
        for _, avg in sorted_tuples:
            total += avg
            total_sqr += avg * avg
        n = len(sorted_tuples)
        mean = total / n
        variance = total_sqr / n - mean * mean
        return mean, variance

    def is_overrepresented(self, cluster: int | str, cluster_reps: list[float], frac: float, n: int) -> bool:
        """
        Check whether cluster is overrepresented wrt. fract.

        :param cluster: selected cluster (int)
        :param cluster_reps: list of selection likelihood of clusters
        :param frac: maximal cluster representation fraction (float)
        :param n: number of instances to select (int)
        :return: True or False
        """
        if cluster_reps[cluster] <= frac:
            cluster_reps[cluster] += 1.0 / n
            return False
        return True

    def get_vector(self, default: float, length: int) -> list[float]:
        """
        Get vector with length and default value.

        :param default: default value of each entry in vector
        :param length: length of vector
        :return: list / vector
        """
        return length * [default]

    def print_samples(self, samples: list[str]) -> None:
        """
        Print sampled instances.

        :param samples: list of instance names
        """
        print(f">> Selected Instances ({len(samples)}):")
        for s in samples:
            print(s)

    def runtime_of_samples(self, samples: list[str]) -> None:
        """
        Print runtimes of sampled instances and aggregation per solver (sum and #timeouts).

        :param samples: list of instance names
        """
        log.info("-" * 30)
        log.info("CSV of runtimes samples")
        sums = self.get_vector(0, len(self.runtime_data[0]))
        timeouts = self.get_vector(0, len(self.runtime_data[0]))
        log.info(",".join(self.solvers) + ",Min,Avg")
        for s in samples:
            times = self._runtime_data_dic[s]
            for index, t in enumerate(times):
                if t == self.cutoff:
                    timeouts[index] += 1
                sums[index] += t
            log.info(f"{s},{','.join(self.to_str_list(times))},{min(times)},{sum(times) / len(times)}")
        log.info("SUM:" + "," + ",".join(self.to_str_list(sums)))
        log.info("Timeouts:" + "," + ",".join(self.to_str_list(timeouts)))

    def features_of_samples(self, samples: list[str]) -> None:
        """
        Print features of sampled instances.

        :param samples: list of instance names
        :return: None
        """
        log.info("-" * 30)
        log.info("CSV of feature samples")
        for s in samples:
            feats = self._feature_data_dic[s]
            log.info(f"{s},{','.join(self.to_str_list(feats))}")

    def to_str_list(self, values: list) -> list[str]:
        """
        Change list of type T to list of type str.

        :param values: list
        :return: list of str
        """
        return [str(v) for v in values]

    def print_stats(self) -> None:
        """
        Print some final stats of sampling.

        :return: None
        """
        log.info("-" * 30)
        log.info("Cluster Distribution:")
        log.info("")
        log.info("Complete Set")
        cluster_dist: dict[int | str, int] = {}
        for cluster in self._clusters.values():
            cluster_dist[cluster] = cluster_dist.get(cluster, 0) + 1
        log.info(cluster_dist)
        log.info("")
        log.info("Selected Set")
        cluster_dist = {}
        for inst in self.samples:
            cluster = self._clusters[inst]
            cluster_dist[cluster] = cluster_dist.get(cluster, 0) + 1
        log.info(cluster_dist)
