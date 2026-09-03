"""
Test cases for ZNormalizer class.
"""

from unittest import TestCase, mock

from setselector import Point
from setselector.zscore_normalizer import ZNormalizer


class TestZNormalizer(TestCase):
    """
    Test cases for the ZNormalizer class.
    """

    def test_init(self) -> None:
        """
        Test class initialization.
        """
        points = [mock.Mock()]
        normalizer = ZNormalizer(points)
        self.assertListEqual(normalizer.feature_data, points)
        self.assertListEqual(normalizer.means, [])
        self.assertListEqual(normalizer.variances, [])

    def test_normalize_features(self) -> None:
        """
        Test the normalization of features.
        """
        points = [mock.Mock()]
        normalizer = ZNormalizer(points)
        coordinates = [[1.0]]
        refs = ["ref"]
        normalizer.points_to_arr = mock.Mock(return_value=(coordinates, refs))
        normalizer.transpose_matrix = mock.Mock(side_effect=["transposed", "normed_transposed"])
        normalizer.get_statistics = mock.Mock(return_value=([1.0], [2.0]))
        normalizer.normalize = mock.Mock(return_value="normed")
        normalizer.arr_to_points = mock.Mock(return_value=points)

        result = normalizer.normalize_features()

        normalizer.points_to_arr.assert_called_once_with(points)
        normalizer.transpose_matrix.assert_has_calls([mock.call(coordinates), mock.call("normed")])
        normalizer.get_statistics.assert_called_once_with("transposed")
        normalizer.normalize.assert_called_once_with("transposed")
        normalizer.arr_to_points.assert_called_once_with("normed_transposed", refs)
        self.assertEqual(normalizer.means, [1.0])
        self.assertEqual(normalizer.variances, [2.0])
        self.assertEqual(normalizer.feature_data, points)
        self.assertEqual(result, points)

    def test_points_to_arr(self) -> None:
        """
        Test conversion of points to array representation.
        """
        point = Point([1.0, 2.0], "ref")
        normalizer = ZNormalizer([point])
        coordinates, refs = normalizer.points_to_arr(normalizer.feature_data)
        self.assertListEqual(coordinates, [[1.0, 2.0]])
        self.assertListEqual(refs, ["ref"])
        self.assertListEqual(normalizer.feature_data, [point])

    def test_arr_to_points(self) -> None:
        """
        Test conversion of array representation back to points.
        """
        normalizer = ZNormalizer([])
        coordinates = [[1.0, 2.0]]
        refs = ["ref"]
        points = normalizer.arr_to_points(coordinates, refs)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].coords, [1.0, 2.0])
        self.assertEqual(points[0].reference, "ref")
        self.assertListEqual(normalizer.feature_data, [])

    def test_transpose_matrix(self) -> None:
        """
        Test the transposition of a matrix.
        """
        normalizer = ZNormalizer([])
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        transposed = normalizer.transpose_matrix(matrix)
        self.assertListEqual(transposed, [[1.0, 3.0], [2.0, 4.0]])

    def test_get_statistics(self) -> None:
        """
        Test the computation of per-row mean and variance.
        """
        normalizer = ZNormalizer([])
        matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        means, variances = normalizer.get_statistics(matrix)
        self.assertListEqual(means, [2.0, 5.0])
        self.assertEqual(len(variances), 2)
        for variance in variances:
            self.assertAlmostEqual(variance, 2.0 / 3.0)

    def test_normalize(self) -> None:
        """
        Test the normalization of a matrix using stored means and variances.
        """
        normalizer = ZNormalizer([])
        normalizer.means = [2.0, 5.0]
        normalizer.variances = [4.0, 4.0]
        matrix = [[0.0, 2.0, 4.0], [3.0, 5.0, 7.0]]
        result = normalizer.normalize(matrix)
        self.assertListEqual(result, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])

    def test_normalize_vector(self) -> None:
        """
        Test the normalization of a single feature vector using stored means and variances.
        """
        normalizer = ZNormalizer([])
        normalizer.means = [2.0, 5.0]
        normalizer.variances = [4.0, 4.0]
        vector = [0.0, 3.0]
        result = normalizer.normalize_vector(vector)
        self.assertListEqual(result, [-1.0, -1.0])
