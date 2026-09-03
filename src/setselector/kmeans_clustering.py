"""
K-means clustering of instance feature vectors.

Author: pandoricweb http://pandoricweb.tumblr.com/post/8646701677/python-implementation-of-the-k-means-clustering
Modified by: Marius Lindauer
Date: 30th May 2012
Modified by: Tom Schmidt
Date: 1st September 2026
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
from collections.abc import Iterable

from . import Cluster, Point, get_distance
from .utils.logging import get_logger
from .zscore_normalizer import ZNormalizer

log = get_logger("selector")


def kmeans(points: list[Point], k: int, cutoff: float, max_its: int) -> list[Cluster]:
    """
    Run k-means clustering until convergence or the iteration limit is reached.

    :param points: The points to cluster.
    :param k: The number of clusters.
    :param cutoff: The convergence threshold for centroid movement.
    :param max_its: The maximum number of iterations.
    :return: The final clusters.
    """
    initial = random.sample(points, k)
    clusters = [Cluster([p]) for p in initial]
    iterations = 0
    while iterations <= max_its:
        iterations += 1
        lists: list[list[Point]] = [[] for _ in clusters]
        for p in points:
            distances = [get_distance(p, c.centroid) for c in clusters]
            index = distances.index(min(distances))
            lists[index].append(p)
        biggest_shift = 0.0
        for i, clu in enumerate(clusters):
            shift = clu.update(lists[i])
            biggest_shift = max(biggest_shift, shift)
        if biggest_shift < cutoff:
            break
    return clusters


def cluster(points: list[Point], k: int) -> tuple[list[Cluster], float]:
    """
    Run a single k-means clustering and return the clusters with their overall quality.

    :param points: The points to cluster.
    :param k: The number of clusters.
    :return: The clusters and their overall quality.
    """
    cutoff = 2
    max_iterations = 1000
    clusters = kmeans(points, k, cutoff, max_iterations)

    log.debug("Quality: ")
    quals = 0.0
    for i, c in enumerate(clusters):
        qual = c.get_quality()
        log.debug(f"Cluster {i}: {qual} \t Mass : {len(c.points)}")
        quals += qual * len(c.points)
    overall = quals / len(points)
    log.debug(f"Overall : {overall}")
    return clusters, overall


def parse_features(feature_file: str) -> dict[str, Point]:
    """
    Parse a csv feature file into a mapping of instance name to Point.

    :param feature_file: The path to the feature CSV file.
    :return: Points indexed by instance name.
    """
    points: dict[str, Point] = {}
    n_feats = -1
    with open(feature_file, encoding="utf-8") as fh:
        _ = fh.readline()
        for line in fh:
            parts = line.split(",")
            inst_name = parts.pop(0)
            values = []
            for value in parts:
                with contextlib.suppress(ValueError):
                    values.append(float(value))
            if values and (n_feats == -1 or len(values) == n_feats):
                if inst_name in points:
                    log.warning(f"Warning Overwrite: duplication of feature data for {inst_name}\n")
                points[inst_name] = Point(values, inst_name)
                if n_feats == -1:
                    n_feats = len(values)
            else:
                log.warning(f"WARNING: {inst_name} has the wrong number of dimensions\n")
                log.warning(f"{values}\n")
    return points


def execute_clustering(points: list[Point], reps: int, k: int) -> list[Cluster]:
    """
    Run several repetitions of k-means and keep the best result.

    :param points: The points to cluster.
    :param reps: The number of clustering repetitions.
    :param k: The number of clusters.
    :return: The highest-quality clustering found.
    """
    best_clusters: list[Cluster] | None = None
    best_qual = float("inf")
    for _ in range(reps):
        clusters, qual = cluster(points, k)
        if qual < best_qual:
            best_clusters = clusters
            best_qual = qual
    assert best_clusters is not None
    log.info(f"Best Quality: {best_qual}")
    for i, c in enumerate(best_clusters):
        log.info(f"Cluster {i} : {c.get_quality()}\t Mass : {len(c.points)}")
    return best_clusters


def get_n_parts(points: list[Point], folds: int) -> list[list[Point]]:
    """
    Split points into (roughly) `folds` random partitions.

    :param points: The points to partition.
    :param folds: The number of partitions.
    :return: Random point partitions.
    """
    length = len(points)
    index = 0
    points_parts: list[list[Point]] = []
    internal_list: list[Point] = []
    part_index = 1
    threshold = length / folds
    points_copy = points[:]
    while points_copy:
        rand_index = random.randint(0, len(points_copy) - 1)
        point = points_copy.pop(rand_index)
        internal_list.append(point)
        if index >= threshold:
            points_parts.append(internal_list)
            internal_list = []
            length = length - index
            threshold = length / (folds - part_index)
            part_index += 1
            index = 0
        index += 1
    points_parts.append(internal_list)
    return points_parts


def join_folds(parts: Iterable[list[Point]], exclude: int) -> list[Point]:
    """
    Concatenate all but the excluded fold.

    :param parts: The folds to concatenate.
    :param exclude: The index of the fold to omit.
    :return: Points from every non-excluded fold.
    """
    joined: list[Point] = []
    for index, part in enumerate(parts):
        if index != exclude:
            joined.extend(part)
    return joined


def do_cluster(
    seed: int,
    feature: str | dict[str, list[float]],
    reps: int,
    clus: int,
    find_k: int,
    read_in: bool,
) -> list[Cluster]:
    """
    Cluster instances given either a feature csv file or a feature dictionary.

    :param seed: The random seed used for clustering.
    :param feature: A feature CSV path or instance-to-feature mapping.
    :param reps: The number of clustering repetitions.
    :param clus: The fixed number of clusters when not selecting it automatically.
    :param find_k: The number of folds used to select the cluster count; non-positive uses `clus`.
    :param read_in: Whether to read features from a CSV file.
    :return: The resulting clusters.
    """
    random.seed(seed)
    if read_in:
        assert isinstance(feature, str)
        points = list(parse_features(feature).values())
    else:
        assert isinstance(feature, dict)
        points = [Point(feats, inst) for inst, feats in feature.items()]
    if len(points) < 2:
        raise ValueError("at least two instances are required for clustering")
    norma = ZNormalizer(points)
    points = norma.normalize_features()

    if find_k > 0:
        points_parts = get_n_parts(points, find_k)

        all_min_dists = []
        for k in range(2, int(math.sqrt(len(points) / 2))):
            sum_dist = 0.0
            for i in range(find_k):
                training = join_folds(points_parts, i)
                clusters = execute_clustering(training, max(reps // 10, 1), k)
                test = points_parts[i]
                for datum in test:
                    sum_dist += min(get_distance(datum, c.centroid) for c in clusters)
            all_min_dists.append(sum_dist)
            if len(all_min_dists) > 1 and sum_dist > all_min_dists[-2]:
                break

        best_k = all_min_dists.index(min(all_min_dists)) + 2 if all_min_dists else 2
        log.info(f"Dists: {all_min_dists}")
        log.info(f"Best K: {best_k}")
        clusters = execute_clustering(points, reps, best_k)
    else:
        clusters = execute_clustering(points, reps, clus)

    return clusters


if __name__ == "__main__":  # nocoverage
    parser = argparse.ArgumentParser()
    req_group = parser.add_argument_group("Options")
    req_group.add_argument("--csv", dest="csv", action="store", required=True, help="csv file with features")
    req_group.add_argument("--k", dest="k", action="store", required=False, type=int, help="number of clusters")
    req_group.add_argument("--r", dest="r", action="store", required=True, type=int, help="repetitions")
    req_group.add_argument("--s", dest="s", action="store", default=1243, required=True, type=int, help="random seed")
    req_group.add_argument(
        "--findK",
        dest="findK",
        action="store",
        default=-1,
        type=int,
        help="use cross fold validation to find the number of clusters (k)",
    )

    args = parser.parse_args()

    do_cluster(args.s, args.csv, args.r, args.k, args.findK, True)
