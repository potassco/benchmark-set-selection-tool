"""
Test cases for kmeans clustering.
"""

from unittest import TestCase, mock

from setselector import Cluster, Point
from setselector.kmeans_clustering import (
    cluster,
    do_cluster,
    execute_clustering,
    get_n_parts,
    join_folds,
    kmeans,
    parse_features,
)


class TestKMeansClustering(TestCase):
    """
    Test cases for kmeans clustering.
    """

    def test_kmeans(self) -> None:
        """
        Test the kmeans function with a simple set of points.
        """
        points = [Point([1.0, 2.0]), Point([3.0, 4.0]), Point([5.0, 6.0]), Point([7.0, 8.0])]
        k = 2
        cutoff = 0.01
        max_its = 100
        with mock.patch("random.sample", return_value=points[:k]):
            clusters = kmeans(points, k, cutoff, max_its)
        self.assertEqual(len(clusters), k)
        total_points = sum(len(c.points) for c in clusters)
        self.assertEqual(total_points, len(points))

    def test_cluster(self) -> None:
        """
        Test cluster function.
        """
        points = [
            Point([0.0]),
            Point([2.0]),
            Point([10.0]),
            Point([14.0]),
        ]
        expected_clusters = [
            Cluster(points[:2]),
            Cluster(points[2:]),
        ]

        with mock.patch(
            "setselector.kmeans_clustering.kmeans",
            return_value=expected_clusters,
        ) as mocked_kmeans:
            clusters, overall_quality = cluster(points, 2)

        mocked_kmeans.assert_called_once_with(points, 2, 2, 1000)
        self.assertIs(clusters, expected_clusters)
        self.assertAlmostEqual(overall_quality, 1.5)

    def test_parse_features(self) -> None:
        """
        Test parse_features, valid, duplicate, and invalid rows.
        """
        contents = "header,f1,f2\nfirst,1.0,2.0\ninvalid,abc\nfirst,4.0,5.0\nempty\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=contents)):
            points = parse_features("features.csv")

        self.assertEqual(list(points), ["first"])
        self.assertEqual(points["first"].coords, [4.0, 5.0])
        self.assertEqual(points["first"].reference, "first")

    def test_execute_clustering(self) -> None:
        """
        Test execute_clustering function.
        """
        points = [Point([0.0]), Point([1.0])]
        other_clusters = [Cluster([points[0]])]
        best_clusters = [Cluster([points[1]])]
        with mock.patch(
            "setselector.kmeans_clustering.cluster",
            side_effect=[(other_clusters, 3.0), (best_clusters, 1.0), (other_clusters, 2.0)],
        ) as mocked_cluster:
            clusters = execute_clustering(points, 3, 1)

        self.assertIs(clusters, best_clusters)
        self.assertEqual(mocked_cluster.call_count, 3)
        mocked_cluster.assert_called_with(points, 1)

    def test_get_n_parts(self) -> None:
        """
        Test get_n_parts function.
        """
        points = [Point([float(index)], str(index)) for index in range(4)]
        with mock.patch("setselector.kmeans_clustering.random.randint", return_value=0):
            parts = get_n_parts(points, 2)

        self.assertEqual(len(parts), 2)
        self.assertEqual([point.reference for part in parts for point in part], ["0", "1", "2", "3"])
        self.assertEqual(sum(len(part) for part in parts), len(points))

    def test_join_folds(self) -> None:
        """
        Test join_folds function.
        """
        parts = [[Point([0.0])], [Point([1.0])], [Point([2.0])]]

        joined = join_folds(parts, 1)

        self.assertEqual(joined, [parts[0][0], parts[2][0]])

    def test_do_cluster(self) -> None:
        """
        Test do_cluster function.
        """
        feature_data = {"first": [1.0], "second": [2.0]}
        normalized_points = [Point([float(index)], str(index)) for index in range(32)]
        expected_clusters = [Cluster([normalized_points[0]])]
        with (
            mock.patch("setselector.kmeans_clustering.ZNormalizer") as mocked_normalizer,
            mock.patch(
                "setselector.kmeans_clustering.execute_clustering", return_value=expected_clusters
            ) as mocked_execute,
            mock.patch(
                "setselector.kmeans_clustering.get_n_parts",
                return_value=[normalized_points[:16], normalized_points[16:]],
            ),
            mock.patch("setselector.kmeans_clustering.get_distance", side_effect=[0.0] * 32 + [1.0] * 32),
        ):
            mocked_normalizer.return_value.normalize_features.return_value = normalized_points

            clusters = do_cluster(123, feature_data, 5, 2, 0, False)
            selected_clusters = do_cluster(123, feature_data, 5, 2, 2, False)

        normalizer_points = mocked_normalizer.call_args.args[0]
        self.assertEqual([point.coords for point in normalizer_points], [[1.0], [2.0]])
        self.assertEqual([point.reference for point in normalizer_points], ["first", "second"])
        self.assertEqual(mocked_execute.call_args_list[-1], mock.call(normalized_points, 5, 2))
        self.assertIs(clusters, expected_clusters)
        self.assertIs(selected_clusters, expected_clusters)

        with (
            mock.patch(
                "setselector.kmeans_clustering.parse_features",
                return_value={"first": Point([1.0], "first"), "second": Point([2.0], "second")},
            ),
            mock.patch("setselector.kmeans_clustering.ZNormalizer") as mocked_file_normalizer,
            mock.patch("setselector.kmeans_clustering.execute_clustering", return_value=expected_clusters),
        ):
            mocked_file_normalizer.return_value.normalize_features.return_value = normalized_points[:2]
            self.assertIs(do_cluster(123, "file.csv", 5, 2, 0, True), expected_clusters)

        with self.assertRaisesRegex(ValueError, "at least two instances"):
            do_cluster(123, {"only": [1.0]}, 5, 2, 0, False)
