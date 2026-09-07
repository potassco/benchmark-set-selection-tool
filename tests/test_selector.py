"""
Test cases for the selector module.
"""

from unittest import TestCase, mock

from setselector import Cluster, Point
from setselector.selector import Selector
from setselector.utils.logging import get_logger

log = get_logger("selector")


class TestSelector(TestCase):
    """
    Test cases for the selector module.
    """

    def test_init(self) -> None:
        """
        Test selector initialization.
        """
        selector = Selector(10)

        self.assertListEqual(selector.solvers, [])
        self.assertDictEqual(selector._feature_data_dic, {})
        self.assertDictEqual(selector._runtime_data_dic, {})
        self.assertListEqual(selector.runtime_data, [])
        self.assertListEqual(selector.feature_data, [])
        self.assertDictEqual(selector._clusters, {})
        self.assertListEqual(selector.samples, [])
        self.assertEqual(selector.cutoff, 10)
        self.assertEqual(selector._n_clusters, 10)

    def test_parse_features(self) -> None:
        """
        Test parsing valid, duplicate, invalid, and inaccessible feature files.
        """
        contents = "instance,f1,f2\nfirst,1,-2\ninvalid,nope\nfirst,3,4\nempty\n"
        selector = Selector(10)
        with mock.patch("builtins.open", mock.mock_open(read_data=contents)):
            self.assertTrue(selector.parse_features("features.csv"))

        self.assertEqual(selector._feature_data_dic, {"first": [3.0, 4.0]})

        with mock.patch("builtins.open", side_effect=OSError):
            self.assertFalse(selector.parse_features("missing.csv"))

    def test_parse_runtimes(self) -> None:
        """
        Test parsing valid, duplicate, invalid, and inaccessible runtime files.
        """
        contents = "instance,solver1,solver2\nfirst,1,20\ninvalid,nope\nfirst,3,4\nempty\n"
        selector = Selector(10)
        with mock.patch("builtins.open", mock.mock_open(read_data=contents)):
            self.assertTrue(selector.parse_runtimes("times.csv"))

        self.assertEqual(selector.solvers, ["instance", "solver1", "solver2"])
        self.assertEqual(selector._runtime_data_dic, {"first": [3.0, 4.0]})

        with mock.patch("builtins.open", side_effect=OSError):
            self.assertFalse(selector.parse_runtimes("missing.csv"))

    def test_random_test_training_split(self) -> None:
        """
        Test that half the runtime instances and corresponding features are removed.
        """
        selector = Selector(10)
        selector._runtime_data_dic = {"first": [1.0], "second": [2.0]}
        selector._feature_data_dic = {"first": [1.0], "second": [2.0]}

        selector.random_test_training_split()

        self.assertEqual(len(selector._runtime_data_dic), 1)
        self.assertDictEqual(selector._runtime_data_dic, selector._feature_data_dic)

    def test_join_times_features(self) -> None:
        """
        Test joining usable data and marking rejected instances.
        """
        selector = Selector(10)
        selector._runtime_data_dic = {
            "hard": [10.0],
            "missing": [1.0],
            "valid": [2.0],
            "wrong_length": [3.0],
            "zero": [4.0],
            "nan": [5.0],
        }
        selector._feature_data_dic = {
            "valid": [1.0, 2.0],
            "wrong_length": [1.0],
            "zero": [0.0, 0.0],
            "nan": [float("nan"), 1.0],
        }

        selector.join_times_features()

        self.assertEqual(selector._runtime_data_dic, {"valid": [2.0]})
        self.assertEqual(selector._feature_data_dic, {"valid": [1.0, 2.0]})
        self.assertEqual(selector.runtime_data, [[2.0]])
        self.assertEqual(selector.feature_data, [[1.0, 2.0]])
        self.assertEqual(selector._clusters, {"hard": "h", "missing": "f", "wrong_length": "f", "nan": "f"})

    def test_clustering(self) -> None:
        """
        Test assigning cluster indexes from k-means results.
        """
        selector = Selector(10)
        clusters = [Cluster([Point([1.0], "first")]), Cluster([Point([2.0], "second")])]
        with mock.patch("setselector.selector.kmeans_clustering.do_cluster", return_value=clusters) as do_cluster:
            selector.clustering(3)

        do_cluster.assert_called_once_with(1234, {}, 3, -1, 10, False)
        self.assertEqual(selector._clusters, {"first": 0, "second": 1})
        self.assertEqual(selector._n_clusters, 3)

    def test_select(self) -> None:
        """
        Test selecting samples for every supported distribution and invalid values.
        """
        selector = Selector(10)
        selector._clusters = {"first": 0, "rejected": "f"}
        selector._n_clusters = 1

        for distribution, random_method in (
            ("gauss", "gauss"),
            ("uni", "random"),
            ("exp", "expovariate"),
            ("log", "lognormvariate"),
        ):
            with (
                self.subTest(distribution=distribution),
                mock.patch(f"setselector.selector.random.{random_method}", return_value=1.0) as mock_random,
                mock.patch.object(selector, "sort_inst", return_value=[("first", 1.0)]),
                mock.patch.object(selector, "get_runtime_statistics", return_value=(1.0, 1.0)),
                mock.patch.object(selector, "find_nearest", return_value=("first", 1.0)),
                mock.patch.object(selector, "is_overrepresented", return_value=False),
            ):
                self.assertEqual(selector.select(1, 1.0, "avg", distribution), ["first"])
                mock_random.assert_called_once()

        with (
            mock.patch("setselector.selector.random.gauss", return_value=1.0),
            mock.patch.object(selector, "sort_inst", return_value=[("rejected", 1.0), ("first", 1.0)]),
            mock.patch.object(selector, "get_runtime_statistics", return_value=(1.0, 1.0)),
            mock.patch.object(selector, "find_nearest", side_effect=[("rejected", 1.0), ("first", 1.0)]),
            mock.patch.object(selector, "is_overrepresented", return_value=False),
        ):
            self.assertEqual(selector.select(1, 1.0, "avg", "gauss"), ["first"])
        with (
            mock.patch("setselector.selector.random.gauss", return_value=1.0),
            mock.patch.object(selector, "sort_inst", return_value=[("first", 1.0), ("first", 1.0)]),
            mock.patch.object(selector, "get_runtime_statistics", return_value=(1.0, 1.0)),
            mock.patch.object(selector, "find_nearest", side_effect=[("first", 1.0), ("first", 1.0)]),
            mock.patch.object(selector, "is_overrepresented", side_effect=[True, False]),
        ):
            self.assertEqual(selector.select(1, 1.0, "avg", "gauss"), ["first"])
        with (
            mock.patch.object(selector, "sort_inst", return_value=[("first", 1.0), ("first", 1.0)]),
            mock.patch.object(selector, "get_runtime_statistics", return_value=(1.0, 1.0)),
            mock.patch.object(selector, "find_nearest", side_effect=[("first", 1.0), ("first", 1.0)]),
            mock.patch.object(selector, "is_overrepresented", side_effect=[True, False]),
            self.assertRaisesRegex(ValueError, "Unknown distribution"),
        ):
            selector.select(1, 1.0, "avg", "unknown")

    def test_remove_too_easy(self) -> None:
        """
        Test removal and marking of too-easy instances.
        """
        selector = Selector(10)
        selector._runtime_data_dic = {"easy": [1.0, 2.0], "hard": [1.0, 8.0]}
        selector._feature_data_dic = {"easy": [1.0], "hard": [2.0]}

        selector.remove_too_easy(5.0, 0.5)

        self.assertEqual(selector._runtime_data_dic, {"hard": [1.0, 8.0]})
        self.assertEqual(selector._feature_data_dic, {"hard": [2.0]})
        self.assertEqual(selector._clusters, {"easy": "e"})

    def test_sort_inst(self) -> None:
        """
        Test every aggregation mode and invalid aggregation names.
        """
        selector = Selector(10)
        selector._runtime_data_dic = {"first": [3.0, 1.0], "second": [2.0, 4.0]}

        self.assertEqual(
            selector.sort_inst("ind"), [("first,2", 1.0), ("second,1", 2.0), ("first,1", 3.0), ("second,2", 4.0)]
        )
        self.assertEqual(selector.sort_inst("avg"), [("first", 2.0), ("second", 3.0)])
        self.assertEqual(selector.sort_inst("min"), [("first", 1.0), ("second", 2.0)])
        with self.assertRaisesRegex(ValueError, "Unknown aggregation"):
            selector.sort_inst("unknown")

    def test_find_nearest(self) -> None:
        """
        Test all nearest-value choices and removal from the input list.
        """
        selector = Selector(10)
        values = [("first", 1.0), ("second", 3.0)]
        self.assertEqual(selector.find_nearest(0.0, values), ("first", 1.0))
        self.assertEqual(values, [("second", 3.0)])

        values = [("first", 1.0), ("second", 3.0)]
        self.assertEqual(selector.find_nearest(2.5, values), ("second", 3.0))

        values = [("first", 1.0), ("second", 3.0)]
        self.assertEqual(selector.find_nearest(1.5, values), ("first", 1.0))
        self.assertEqual(selector.find_nearest(5.0, [("first", 1.0)]), ("first", 1.0))

    def test_get_runtime_statistics(self) -> None:
        """
        Test the computation of runtime statistics.
        """
        selector = Selector(10)
        self.assertEqual(selector.get_runtime_statistics([("first", 1.0), ("second", 3.0)]), (2.0, 1.0))

    def test_is_overrepresented(self) -> None:
        """
        Test the check for overrepresented clusters.
        """
        selector = Selector(10)
        self.assertFalse(selector.is_overrepresented(0, [0.0], 0.5, 2))
        self.assertTrue(selector.is_overrepresented(0, [1.0], 0.5, 2))

    def test_get_vector(self) -> None:
        """
        Test the creation of vectors with repeated values.
        """
        selector = Selector(10)
        self.assertEqual(selector.get_vector(2.0, 3), [2.0, 2.0, 2.0])

    def test_print_samples(self) -> None:
        """
        Test the printing of sampled instances.
        """
        selector = Selector(10)
        samples = ["first", "second"]
        with mock.patch("builtins.print") as print_mock:
            selector.print_samples(samples)
        print_mock.assert_has_calls([mock.call(">> Selected Instances (2):"), mock.call("first"), mock.call("second")])

    def test_runtime_of_samples(self) -> None:
        """
        Test the printing of runtimes of sampled instances.
        """
        selector = Selector(10)
        selector.solvers = ["solver1", "solver2"]
        selector.runtime_data = [[1.0, 2.0]]
        selector._runtime_data_dic = {"first": [10.0, 20.0], "second": [2.0, 4.0]}
        selector.cutoff = 20.0
        samples = ["first", "second"]
        with mock.patch.object(log, "info") as log_mock:
            selector.runtime_of_samples(samples)
        self.assertListEqual(
            log_mock.call_args_list,
            [
                mock.call("------------------------------"),
                mock.call("CSV of runtimes samples"),
                mock.call("solver1,solver2,Min,Avg"),
                mock.call("first,10.0,20.0,10.0,15.0"),
                mock.call("second,2.0,4.0,2.0,3.0"),
                mock.call("SUM:12.0,24.0"),
                mock.call("Timeouts:0,1"),
            ],
        )

    def test_features_of_samples(self) -> None:
        """
        Test the printing of features of sampled instances.
        """
        selector = Selector(10)
        selector._feature_data_dic = {"first": [1.0, 2.0], "second": [3.0, 4.0]}
        samples = ["first", "second"]
        with mock.patch.object(log, "info") as log_mock:
            selector.features_of_samples(samples)
        log_mock.assert_any_call("CSV of feature samples")

    def test_to_str_list(self) -> None:
        """
        Test the conversion of a list of values to a list of strings.
        """
        selector = Selector(10)
        self.assertEqual(selector.to_str_list([1, 2.0, "three"]), ["1", "2.0", "three"])

    def test_print_stats(self) -> None:
        """
        Test the printing of final statistics of sampling.
        """
        selector = Selector(10)
        selector._clusters = {"first": 0, "second": 0, "third": 1}
        selector.samples = ["first", "second"]
        with mock.patch.object(log, "info") as log_mock:
            selector.print_stats()
        self.assertListEqual(
            log_mock.call_args_list,
            [
                mock.call("------------------------------"),
                mock.call("Cluster Distribution:"),
                mock.call(""),
                mock.call("Complete Set"),
                mock.call({0: 2, 1: 1}),
                mock.call(""),
                mock.call("Selected Set"),
                mock.call({0: 2}),
            ],
        )
