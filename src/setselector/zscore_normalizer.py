"""
Z-score (mean/variance) normalization of feature vectors.

Author: Marius Lindauer
Modified by: Tom Schmidt
Date: 1st September 2026
"""

import math

from . import Point
from .utils.logging import get_logger

log = get_logger("selector")


class ZNormalizer:
    """
    Normalizes feature vectors of `Point`s to zero mean and unit variance.
    """

    def __init__(self, feature_data: list[Point]) -> None:
        """
        Initialize the normalizer with the given feature data.

        :param feature_data: The points whose features will be normalized.
        """
        self.feature_data = feature_data
        self.means: list[float] = []
        self.variances: list[float] = []

    def normalize_features(self) -> list[Point]:
        """
        Normalize the stored points and return them.

        :return: The normalized points.
        """
        coordinates, refs = self.points_to_arr(self.feature_data)
        transposed_features = self.transpose_matrix(coordinates)
        self.means, self.variances = self.get_statistics(transposed_features)
        trans_normed_features = self.normalize(transposed_features)
        normalized_coordinates = self.transpose_matrix(trans_normed_features)
        self.feature_data = self.arr_to_points(normalized_coordinates, refs)
        return self.feature_data

    def points_to_arr(self, points: list[Point]) -> tuple[list[list[float]], list[object]]:
        """
        Convert points into coordinate and reference lists.

        :param points: The points to convert.
        :return: Coordinate rows and their corresponding references.
        """
        return [point.coords for point in points], [point.reference for point in points]

    def arr_to_points(self, coordinates: list[list[float]], refs: list[object]) -> list[Point]:
        """
        Convert coordinate and reference lists back into points.

        :param coordinates: The coordinate rows.
        :param refs: References corresponding to the coordinate rows.
        :return: Points reconstructed from the coordinates and references.
        """
        return [Point(coords, ref) for coords, ref in zip(coordinates, refs, strict=True)]

    def transpose_matrix(self, matrix: list[list[float]]) -> list[list[float]]:
        """
        Transpose a matrix given as a list of rows.

        :param matrix: The matrix to transpose.
        :return: The transposed matrix.
        """
        return [list(column) for column in zip(*matrix, strict=True)]

    def get_statistics(self, matrix: list[list[float]]) -> tuple[list[float], list[float]]:
        """
        Compute per-row mean and variance (floored at 0.001).

        :param matrix: The matrix whose row statistics are computed.
        :return: Per-row means and variances.
        """
        means = []
        variances = []
        for line in matrix:
            n = len(line)
            mean = sum(line) / n
            variance = sum(value * value for value in line) / n - mean * mean
            means.append(mean)
            variances.append(max(variance, 0.001))
        return means, variances

    def normalize(self, matrix: list[list[float]]) -> list[list[float]]:
        """
        Normalize each row of the matrix using the stored means and variances.

        :param matrix: The matrix to normalize.
        :return: The normalized matrix.
        """
        norm_matrix = []
        for index, line in enumerate(matrix):
            mean = self.means[index]
            std = math.sqrt(self.variances[index])
            norm_matrix.append([(float(value) - mean) / std for value in line])
        print(norm_matrix)
        return norm_matrix

    def normalize_vector(self, vector: list[float]) -> list[float]:
        """
        Normalize a single feature vector using the stored means and variances.

        :param vector: The feature vector to normalize.
        :return: The normalized feature vector.
        """
        return [
            (float(value) - self.means[index]) / math.sqrt(self.variances[index]) for index, value in enumerate(vector)
        ]
