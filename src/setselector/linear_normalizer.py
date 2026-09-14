"""
Min/max (linear) normalization of feature vectors.

Author: Marius Lindauer
Modified by: Tom Schmidt
Date: 1st September 2026
"""

from . import Point
from .utils.logging import get_logger

log = get_logger("selector")


class LNormalizer:
    """
    Normalizes feature vectors of `Point`s to the [0, 1] range.
    """

    def __init__(self, feature_data: list[Point]) -> None:
        """
        Initialize the linear normalizer with the given feature data.

        :param feature_data: The points whose features will be normalized.
        """
        self.feature_data = feature_data
        self.mins: list[float] = []
        self.maxs: list[float] = []

    def normalize_features(self) -> list[Point]:
        """
        Normalize the stored points and return them.

        :return: The normalized points.
        """
        coordinates, refs = self.points_to_arr(self.feature_data)
        transposed_features = self.transpose_matrix(coordinates)
        self.maxs, self.mins = self.get_statistics(transposed_features)
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
        Compute per-row max and min values.

        :param matrix: The matrix whose row statistics are computed.
        :return: Per-row maximum and minimum values.
        """
        maxs = [max(line) for line in matrix]
        mins = [min(line) for line in matrix]
        return maxs, mins

    def normalize(self, matrix: list[list[float]]) -> list[list[float]]:
        """
        Normalize each row of the matrix using the stored maxs and mins.

        :param matrix: The matrix to normalize.
        :return: The normalized matrix.
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

        :param vector: The feature vector to normalize.
        :return: The normalized feature vector.
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
