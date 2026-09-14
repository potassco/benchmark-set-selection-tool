"""
Utility classes and functions.
"""

import math
import random
from collections.abc import Sequence
from typing import Any


class Point:
    """
    A feature vector optionally tagged with a reference (e.g. instance name).
    """

    def __init__(self, coords: Sequence[float], reference: Any = None) -> None:  # noqa: ANN401
        """
        Initialize the point with the given coordinates and an optional reference.

        The reference can be used to tag the point (e.g., with an instance name).
        :param coords: The coordinates of the point.
        :param reference: An optional reference for the point.
        """
        self.coords = list(coords)
        self.n = len(self.coords)
        self.reference = reference

    def __repr__(self) -> str:
        """
        Return a string representation of the point, showing its reference.

        :return: The point reference as text.
        """
        return str(self.reference)


class Cluster:
    """
    A cluster of points with a cached centroid.
    """

    def __init__(self, points: list[Point]) -> None:
        """
        Initialize the cluster with the given points and compute its centroid.

        :param points: The points assigned to the cluster.
        """
        if len(points) == 0:
            raise ValueError("ILLEGAL: empty cluster")
        self.points = points
        self.n = points[0].n
        for p in points:
            if p.n != self.n:
                raise ValueError("ILLEGAL: wrong dimensions")
        self.centroid = self.calculate_centroid()

    def __repr__(self) -> str:
        """
        Return a string representation of the cluster, showing its points.

        :return: The cluster points as text.
        """
        return str(self.points)

    def update(self, points: list[Point]) -> float:
        """
        Assign new points to the cluster and return the centroid shift.

        :param points: The points to assign to the cluster.
        :return: The Euclidean distance the centroid moved.
        """
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculate_centroid()
        return get_distance(old_centroid, self.centroid)

    def calculate_centroid(self) -> Point:
        """
        Compute the centroid of the current points.

        :return: The current centroid, or a random point for an empty cluster.
        """
        if self.points:
            centroid_coords = [sum(p.coords[i] for p in self.points) / len(self.points) for i in range(self.n)]
            return Point(centroid_coords)
        return make_random_point(self.n, -1, 1)

    def get_quality(self) -> float:
        """
        Average distance of the points to the centroid.

        :return: The mean distance to the centroid.
        """
        if not self.points:
            return 0.0
        return sum(get_distance(p, self.centroid) for p in self.points) / len(self.points)


def get_distance(a: Point, b: Point) -> float:
    """
    Euclidean distance between two points.

    :param a: The first point.
    :param b: The second point.
    :return: The Euclidean distance between the points.
    """
    if a.n != b.n:
        raise ValueError("ILLEGAL: non comparable points")
    return math.sqrt(sum((a.coords[i] - b.coords[i]) ** 2 for i in range(a.n)))


def make_random_point(n: int, lower: float, upper: float) -> Point:
    """
    Create a point with random coordinates in [lower, upper].

    :param n: The number of coordinates.
    :param lower: The inclusive lower coordinate bound.
    :param upper: The inclusive upper coordinate bound.
    :return: A point with uniformly sampled coordinates.
    """
    return Point([random.uniform(lower, upper) for _ in range(n)])
