"""
Min/max (linear) normalization of feature vectors.

Author: Marius Schneider
Modified by: Tom Schmidt
Date: 1st September 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .utils.logging import get_logger

if TYPE_CHECKING:
    from .clusterKMeans import Point

log = get_logger("selector")

class LNormalizer:
    """
    Normalizes feature vectors of `Point`s to the [0, 1] range.
    """

    def __init__(self, feature_data: list[Point]) -> None:
        self.feature_data: list[Point] | list[list[float]] = feature_data
        self.mins: list[float] = []
        self.maxs: list[float] = []
        self.refs: list[object] = []

    def normalize_features(self) -> list[Point]:
        """
        Normalize the stored points and return them.
        """
        self.points_to_arr()
        transposed_features = self.transpose_matrix(self.feature_data)
        self.maxs, self.mins = self.get_statistics(transposed_features)
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
        Compute per-row max and min values.
        """
        maxs = [max(line) for line in matrix]
        mins = [min(line) for line in matrix]
        return maxs, mins

    def normalize(self, matrix: list[list[float]]) -> list[list[float]]:
        """
        Normalize each row of the matrix using the stored maxs and mins.
        """
        norm_matrix = []
        for index, line in enumerate(matrix):
            maxi = self.maxs[index]
            mini = self.mins[index]
            if maxi == mini:
                norm_matrix.append([0.0 for _ in line])
            else:
                norm_matrix.append([(value - mini) / (maxi - mini) for value in line])
        return norm_matrix

    def normalize_vector(self, vector: list[float]) -> list[float]:
        """
        Normalize a single feature vector using the stored maxs and mins.
        """
        normed_vector = []
        for index, value in enumerate(vector):
            mini = self.mins[index]
            maxi = self.maxs[index]
            if mini != maxi:
                normed_vector.append((float(value) - mini) / (maxi - mini))
            else:
                normed_vector.append(0.0)
        return normed_vector
