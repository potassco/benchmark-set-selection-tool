"""
The command line parser for the project.
"""

from argparse import ArgumentParser
from importlib import metadata
from textwrap import dedent
from typing import Any, cast

from . import logging

__all__ = ["get_parser"]

VERSION = metadata.version("setselector")


def get_parser() -> ArgumentParser:
    """
    Return the parser for command line options.
    """
    parser = ArgumentParser(
        prog="setselector",
        description=dedent("""\
            setselector
            filldescription
            """),
    )
    levels = [
        ("error", logging.ERROR),
        ("warning", logging.WARNING),
        ("info", logging.INFO),
        ("debug", logging.DEBUG),
    ]

    def get(levels: list[tuple[str, int]], name: str) -> int | None:
        for key, val in levels:
            if key == name:
                return val
        return None  # nocoverage

    parser.add_argument(
        "--log",
        default="warning",
        choices=[val for _, val in levels],
        metavar=f"{{{','.join(key for key, _ in levels)}}}",
        help="set log level [%(default)s]",
        type=cast(Any, lambda name: get(levels, name)),
    )

    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {VERSION}")

    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--runtimes", dest="times", action="store", help="runtimes in csv (first col with instance names)"
    )
    input_group.add_argument(
        "--features",
        dest="feats",
        action="store",
        help="instance features in csv (first col with instance names)",
    )
    input_group.add_argument(
        "--eval",
        dest="eval",
        action="store",
        help="evaluation data in xml (produced by benchmark-tool)",
    )

    req_group = parser.add_argument_group("Required Options")
    req_group.add_argument(
        "--cutoff",
        dest="cutoff",
        action="store",
        type=int,
        required=True,
        help="cutoff time (default: %(default)s)",
    )
    req_group.add_argument(
        "--n",
        dest="n",
        action="store",
        type=int,
        required=True,
        help="desired number of instances (default: %(default)s)",
    )

    opt_group = parser.add_argument_group("Optional Options")
    opt_group.add_argument(
        "--reps",
        dest="reps",
        action="store",
        default=100,
        type=int,
        help="repetitions of kmeans clustering (default: %(default)s)",
    )
    opt_group.add_argument(
        "--frac",
        dest="frac",
        action="store",
        default=0.2,
        type=float,
        help="maximum representation of each cluster [0,1] (default: %(default)s)",
    )
    opt_group.add_argument(
        "--easyK",
        dest="easyK",
        action="store",
        default=0.1,
        type=float,
        help="remove too easy instances (solved by all solvers and avg runtimes < k*cutoff) (default: %(default)s)",
    )
    opt_group.add_argument(
        "--aggregate",
        dest="agg",
        action="store",
        default="avg",
        choices=["avg", "min", "ind"],
        help="aggregation of instance runtimes (default: %(default)s)",
    )
    opt_group.add_argument(
        "--dist",
        dest="dist",
        action="store",
        default="gauss",
        choices=["gauss", "uni", "exp", "log"],
        help="sample distribution (default: %(default)s)",
    )
    opt_group.add_argument(
        "--split",
        dest="split",
        action="store_true",
        default=False,
        help="before sampling, split randomly instance set in test and training set",
    )

    return parser
