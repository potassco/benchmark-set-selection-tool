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

    csv_used = args.times is not None or args.feats is not None
    eval_used = args.eval is not None

    if csv_used and eval_used:
        parser.error("Cannot use both CSV input and evaluation data simultaneously.")
    elif not eval_used and (args.times is None or args.feats is None):
        parser.error("Provide either --eval XML or both --runtimes CSV and --features CSV.")

    configure_logging(sys.stderr, args.log, sys.stderr.isatty())

    log = get_logger("main")

    if 0 < args.easyK * args.cutoff < 10.0:
        args.easyK = 10.0 / args.cutoff
        log.info("Adjusted easyK to %f based on cutoff %f", args.easyK, args.cutoff)

    selector = Selector(args.cutoff)
    if csv_used:
        selector.parse_features(args.feats)
        selector.parse_runtimes(args.times)
    else:
        selector.parse_eval(args.eval, args.eval_features)
    if not selector._runtime_data_dic or not any(selector._feature_data_dic.values()):
        parser.error("No usable benchmark data after parsing input. Check input files and options.")

    if args.split:
        selector.random_test_training_split()

    selector.join_times_features()
    if not selector._runtime_data_dic:
        parser.error("No usable benchmark data remains after filtering too hard instances.")
    selector.runtime_of_samples(list(selector._runtime_data_dic.keys()))
    selector.remove_too_easy(args.cutoff, args.easyK)
    if not selector._runtime_data_dic:
        parser.error("No usable benchmark data remains after filtering too easy instances.")
    selector.clustering(args.reps)
    samples = selector.select(args.n, args.frac, args.agg, args.dist)
    selector.print_samples(samples)
    selector.runtime_of_samples(samples)
    selector.features_of_samples(samples)
    selector.print_stats()

    log.debug("done")


if __name__ == "__main__":
    main()
