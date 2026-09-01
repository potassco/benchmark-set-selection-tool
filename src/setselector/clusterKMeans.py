"""
K-means clustering of instance feature vectors.

Author: pandoricweb http://pandoricweb.tumblr.com/post/8646701677/python-implementation-of-the-k-means-clustering
Modified by: Marius Lindauer
Date: 30th May 2012
Modified by: Tom Schmidt
Date: 1st September 2026
"""

from __future__ import annotations
from .utils.logging import get_logger

import argparse
import math
import random
import sys
from collections.abc import Iterable, Sequence
from typing import Any

from .zNormalizer import ZNormalizer

log = get_logger("selector")

class Point:
    """
    A feature vector optionally tagged with a reference (e.g. instance name).
    """

    def __init__(self, coords: Sequence[float], reference: Any = None) -> None:
        self.coords = list(coords)
        self.n = len(self.coords)
        self.reference = reference

    def __repr__(self) -> str:
        return str(self.reference)


class Cluster:
    """
    A cluster of points with a cached centroid.
    """

    def __init__(self, points: list[Point]) -> None:
        if len(points) == 0:
            raise ValueError("ILLEGAL: empty cluster")
        self.points = points
        self.n = points[0].n
        for p in points:
            if p.n != self.n:
                raise ValueError("ILLEGAL: wrong dimensions")
        self.centroid = self.calculate_centroid()

    def __repr__(self) -> str:
        return str(self.points)

    def update(self, points: list[Point]) -> float:
        """
        Assign new points to the cluster and return the centroid shift.
        """
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculate_centroid()
        return get_distance(old_centroid, self.centroid)

    def calculate_centroid(self) -> Point:
        """
        Compute the centroid of the current points.
        """
        if self.points:
            centroid_coords = [
                sum(p.coords[i] for p in self.points) / len(self.points) for i in range(self.n)
            ]
            return Point(centroid_coords)
        return make_random_point(self.n, -1, 1)

    def get_quality(self) -> float:
        """
        Average distance of the points to the centroid.
        """
        if not self.points:
            return 0.0
        return sum(get_distance(p, self.centroid) for p in self.points) / len(self.points)


def kmeans(points: list[Point], k: int, cutoff: float, max_its: int) -> list[Cluster]:
    """
    Run k-means clustering until convergence or the iteration limit is reached.
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


def get_distance(a: Point, b: Point) -> float:
    """
    Euclidean distance between two points.
    """
    if a.n != b.n:
        raise ValueError("ILLEGAL: non comparable points")
    return math.sqrt(sum((a.coords[i] - b.coords[i]) ** 2 for i in range(a.n)))


def make_random_point(n: int, lower: float, upper: float) -> Point:
    """
    Create a point with random coordinates in [lower, upper].
    """
    return Point([random.uniform(lower, upper) for _ in range(n)])


def cluster(points: list[Point], k: int) -> tuple[list[Cluster], float]:
    """
    Run a single k-means clustering and return the clusters with their overall quality.
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
    """
    points: dict[str, Point] = {}
    n_feats = -1
    with open(feature_file, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split(",")
            inst_name = parts.pop(0)
            values = []
            for value in parts:
                try:
                    values.append(float(value))
                except ValueError:
                    pass
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    req_group = parser.add_argument_group("Options")
    req_group.add_argument("--csv", dest="csv", action="store", required=True, help="csv file with features")
    req_group.add_argument("--k", dest="k", action="store", required=False, type=int, help="number of clusters")
    req_group.add_argument("--r", dest="r", action="store", required=True, type=int, help="repetitions")
    req_group.add_argument(
        "--s", dest="s", action="store", default=1243, required=True, type=int, help="random seed"
    )
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

        
