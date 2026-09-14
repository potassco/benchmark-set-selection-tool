"""
Test cases for helper classes.
"""

from unittest import TestCase, mock

from setselector import Cluster, Point, get_distance, make_random_point


class TestPoint(TestCase):
    """
    Test cases for the Point helper class.
    """

    def test_init(self) -> None:
        """
        Test point initialization.
        """
        point = Point((1.0, 2.0), "instance")

        self.assertEqual(point.coords, [1.0, 2.0])
        self.assertEqual(point.n, 2)
        self.assertEqual(point.reference, "instance")

    def test_repr(self) -> None:
        """
        Test point representation.
        """
        point = Point([], "instance")
        with mock.patch("setselector.str", return_value="instance") as string:
            self.assertEqual(repr(point), "instance")
        string.assert_called_once_with("instance")


class TestCluster(TestCase):
    """
    Test cases for the Cluster helper class.
    """

    def test_init(self) -> None:
        """
        Test cluster initialization and invalid inputs.
        """
        first = mock.Mock(n=2)
        second = mock.Mock(n=2)
        with mock.patch.object(Cluster, "calculate_centroid", return_value="centroid") as calculate_centroid:
            cluster = Cluster([first, second])

        self.assertEqual(cluster.points, [first, second])
        self.assertEqual(cluster.n, 2)
        self.assertEqual(cluster.centroid, "centroid")
        calculate_centroid.assert_called_once_with()

        with self.assertRaisesRegex(ValueError, "empty cluster"):
            Cluster([])
        with self.assertRaisesRegex(ValueError, "wrong dimensions"):
            Cluster([first, mock.Mock(n=3)])

    def test_repr(self) -> None:
        """
        Test cluster representation.
        """
        cluster = mock.Mock(points=["first"])
        with mock.patch("setselector.str", return_value="['first']") as string:
            self.assertEqual(Cluster.__repr__(cluster), "['first']")
        string.assert_called_once_with(["first"])

    def test_update(self) -> None:
        """
        Test cluster updates return the centroid shift.
        """
        cluster = mock.Mock(centroid="old")
        cluster.calculate_centroid.return_value = "new"
        with mock.patch("setselector.get_distance", return_value=1.5) as distance:
            self.assertEqual(Cluster.update(cluster, ["point"]), 1.5)

        self.assertEqual(cluster.points, ["point"])
        self.assertEqual(cluster.centroid, "new")
        cluster.calculate_centroid.assert_called_once_with()
        distance.assert_called_once_with("old", "new")

    def test_calculate_centroid(self) -> None:
        """
        Test centroids for populated and empty clusters.
        """
        cluster = mock.Mock(n=2, points=[mock.Mock(coords=[1.0, 3.0]), mock.Mock(coords=[3.0, 5.0])])
        with mock.patch("setselector.Point", return_value="centroid") as point:
            self.assertEqual(Cluster.calculate_centroid(cluster), "centroid")
        point.assert_called_once_with([2.0, 4.0])

        cluster.points = []
        with mock.patch("setselector.make_random_point", return_value="random") as random_point:
            self.assertEqual(Cluster.calculate_centroid(cluster), "random")
        random_point.assert_called_once_with(2, -1, 1)

    def test_get_quality(self) -> None:
        """
        Test populated and empty cluster quality.
        """
        cluster = mock.Mock(points=["first", "second"], centroid="centroid")
        with mock.patch("setselector.get_distance", side_effect=[1.0, 3.0]) as distance:
            self.assertEqual(Cluster.get_quality(cluster), 2.0)
        self.assertEqual(distance.call_args_list, [mock.call("first", "centroid"), mock.call("second", "centroid")])

        cluster.points = []
        self.assertEqual(Cluster.get_quality(cluster), 0.0)


class TestFunctions(TestCase):
    """
    Test cases for helper functions.
    """

    def test_get_distance(self) -> None:
        """
        Test distance calculation and incompatible points.
        """
        first = mock.Mock(n=2, coords=[1.0, 2.0])
        second = mock.Mock(n=2, coords=[4.0, 6.0])
        with mock.patch("setselector.math.sqrt", return_value=5.0) as sqrt:
            self.assertEqual(get_distance(first, second), 5.0)
        sqrt.assert_called_once_with(25.0)

        with self.assertRaisesRegex(ValueError, "non comparable points"):
            get_distance(first, mock.Mock(n=3))

    def test_make_random_point(self) -> None:
        """
        Test random point creation.
        """
        with (
            mock.patch("setselector.random.uniform", side_effect=[0.1, 0.2]) as uniform,
            mock.patch("setselector.Point", return_value="point") as point,
        ):
            self.assertEqual(make_random_point(2, -1.0, 1.0), "point")

        self.assertEqual(uniform.call_args_list, [mock.call(-1.0, 1.0), mock.call(-1.0, 1.0)])
        point.assert_called_once_with([0.1, 0.2])
