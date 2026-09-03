"""
Nox sessions for linting, type checking and testing.
"""

import os

import nox  # type: ignore

nox.options.sessions = "format, lint", "typecheck", "test"
nox.options.default_venv_backend = "uv|virtualenv"

PYTHON_VERSIONS = None
if "GITHUB_ACTIONS" in os.environ:
    PYTHON_VERSIONS = ["3.12", "3.14"]

FILES_TO_BE_CHECKED = [
    "src",
]


@nox.session
def format(session):
    """
    Autoformat source files.

    If argument check is given, only reports changes.
    """
    session.install("-e", ".[format]")
    check = "check" in session.posargs

    autoflake_args = [
        "--in-place",
        "--imports=benchmarktool",
        "--ignore-init-module-imports",
        "--remove-unused-variables",
        "-r",
        "tests",
    ] + FILES_TO_BE_CHECKED
    if check:
        autoflake_args.remove("--in-place")
    session.run("autoflake", *autoflake_args)

    isort_args = ["--profile", "black", "tests"] + FILES_TO_BE_CHECKED
    if check:
        isort_args.insert(0, "--check")
        isort_args.insert(1, "--diff")
    session.run("isort", *isort_args)

    black_args = ["tests"] + FILES_TO_BE_CHECKED
    if check:
        black_args.insert(0, "--check")
        black_args.insert(1, "--diff")
    session.run("black", *black_args)


@nox.session
def dev(session):
    """
    Create a development environment in editable mode.

    Activate it by running `source .nox/dev/bin/activate`.
    """
    session.install("-e", ".[dev]")


@nox.session
def lint(session):
    """
    Run ruff.
    """
    if not session.virtualenv._reused:
        session.install(".[lint]")
    session.run("ruff", "check")
    session.run("ruff", "format", "--check")


@nox.session
def typecheck(session):
    """
    Typecheck the code using mypy.
    """
    if not session.virtualenv._reused:
        session.install(".[typecheck]")
    session.run("ty", "check", *FILES_TO_BE_CHECKED)


@nox.session(python=PYTHON_VERSIONS)
def test(session):
    """
    Run the tests.

    Accepts additional arguments which are passed to the pytest module. This
    can for example be used to selectively run test cases via option `-k`.
    """
    if not session.virtualenv._reused:
        session.install(".[test]")
    if session.posargs:
        session.run("pytest", "-v", *session.posargs)
    else:
        session.run("pytest", "--cov", "-v")
