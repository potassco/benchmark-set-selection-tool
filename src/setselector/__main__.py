"""
The main entry point for the application.
"""

import sys

from .selector import Selector
from .utils.logging import configure_logging, get_logger
from .utils.parser import get_parser


def main() -> None:
    """
    Parse command-line arguments and run benchmark selection.

    :return: None
    """
    parser = get_parser()
    args = parser.parse_args()
    configure_logging(sys.stderr, args.log, sys.stderr.isatty())

    log = get_logger("main")

    if 0 < args.easyK * args.cutoff < 10.0:
        args.easyK = 10.0 / args.cutoff
        log.info("Adjusted easyK to %f based on cutoff %f", args.easyK, args.cutoff)

    selector = Selector(args.cutoff)
    selector.parse_features(args.feats)
    selector.parse_runtimes(args.times)

    if args.split:
        selector.random_test_training_split()

    selector.join_times_features()
    selector.runtime_of_samples(list(selector._runtime_data_dic.keys()))
    selector.remove_too_easy(args.cutoff, args.easyK)
    selector.clustering(args.reps)
    samples = selector.select(args.n, args.frac, args.agg, args.dist)
    selector.print_samples(samples)
    selector.runtime_of_samples(samples)
    selector.features_of_samples(samples)
    selector.print_stats()

    log.debug("done")


if __name__ == "__main__":
    main()
