"""
Z-score (mean/variance) normalization of feature vectors.

Author: Marius Lindauer
Modified by: Tom Schmidt
Date: 1st September 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .utils.logging import get_logger

if TYPE_CHECKING:
    from .clusterKMeans import Point



log = get_logger("selector")

class ZNormalizer:
    """
    Normalizes feature vectors of `Point`s to zero mean and unit variance.
    """

    def __init__(self, feature_data: list[Point]) -> None:
        self.feature_data: list[Point] | list[list[float]] = feature_data
        self.means: list[float] = []
        self.variances: list[float] = []
        self.refs: list[object] = []

    def normalize_features(self) -> list[Point]:
        """
        Normalize the stored points and return them.
        """
        self.points_to_arr()
        transposed_features = self.transpose_matrix(self.feature_data)
        self.means, self.variances = self.get_statistics(transposed_features)
        trans_normed_features = self.normalize(transposed_features)
        self.feature_data = self.transpose_matrix(trans_normed_features)
        return self.arr_to_points()

    def points_to_arr(self) -> None:
        """
        Convert the stored points into plain coordinate lists.
        """
        arr = []
        for p in self.feature_data:
            arr.append(p.coords)
            self.refs.append(p.reference)
        self.feature_data = arr

    def arr_to_points(self) -> list[Point]:
        """
        Convert the stored coordinate lists back into points.
        """
        from .clusterKMeans import Point

        points = []
        for a in self.feature_data:
            ref = self.refs.pop(0)
            points.append(Point(a, ref))
        self.feature_data = points
        return points

    def transpose_matrix(self, matrix: list[list[float]]) -> list[list[float]]:
        """
        Transpose a matrix given as a list of rows.
        """
        return [list(column) for column in zip(*matrix)]

    def get_statistics(self, matrix: list[list[float]]) -> tuple[list[float], list[float]]:
        """
        Compute per-row mean and variance (floored at 0.001).
        """
        means = []
        variances = []
        for line in matrix:
            n = len(line)
            mean = sum(line) / n
            variance = sum(value * value for value in line) / n - mean * mean
            means.append(mean)
            variances.append(variance if variance > 0.001 else 0.001)
        return means, variances

    def normalize(self, matrix: list[list[float]]) -> list[list[float]]:
        """
        Normalize each row of the matrix using the stored means and variances.
        """
        norm_matrix = []
        for index, line in enumerate(matrix):
            mean = self.means[index]
            std = math.sqrt(self.variances[index])
            norm_matrix.append([(float(value) - mean) / std for value in line])
        return norm_matrix

    def normalize_vector(self, vector: list[float]) -> list[float]:
        """
        Normalize a single feature vector using the stored means and variances.
        """
        return [
            (float(value) - self.means[index]) / math.sqrt(self.variances[index])
            for index, value in enumerate(vector)
        ]
