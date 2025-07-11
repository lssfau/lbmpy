from __future__ import annotations
from typing import Sequence
from argparse import ArgumentParser

import os
import nox
import subprocess
import re

nox.options.sessions = ["lint", "typecheck"]


def get_cuda_version(session: nox.Session) -> None | tuple[int, ...]:
    query_args = ["nvcc", "--version"]

    try:
        query_result = subprocess.run(query_args, capture_output=True)
    except FileNotFoundError:
        return None

    matches = re.findall(r"release \d+\.\d+", str(query_result.stdout))
    if matches:
        match = matches[0]
        version_string = match.split()[-1]
        try:
            return tuple(int(v) for v in version_string.split("."))
        except ValueError:
            pass

    session.warn("nvcc was found, but I am unable to determine the CUDA version.")
    return None


def install_cupy(
    session: nox.Session, cupy_version: str, skip_if_no_cuda: bool = False
):
    if cupy_version is not None:
        cuda_version = get_cuda_version(session)
        if cuda_version is None or cuda_version[0] not in (11, 12):
            if skip_if_no_cuda:
                session.skip(
                    "No compatible installation of CUDA found - Need either CUDA 11 or 12"
                )
            else:
                session.warn(
                    "Running without cupy: no compatbile installation of CUDA found. Need either CUDA 11 or 12."
                )
                return

        cuda_major = cuda_version[0]
        cupy_package = f"cupy-cuda{cuda_major}x=={cupy_version}"
        session.install(cupy_package)


def check_external_doc_dependencies(session: nox.Session):
    dot_args = ["dot", "--version"]
    try:
        _ = subprocess.run(dot_args, capture_output=True)
    except FileNotFoundError:
        session.error(
            "Unable to build documentation: "
            "Command `dot` from the `graphviz` package (https://www.graphviz.org/) is not available"
        )


def editable_install(session: nox.Session, opts: Sequence[str] = ()):
    if opts:
        opts_str = "[" + ",".join(opts) + "]"
    else:
        opts_str = ""
    session.install("-e", f".{opts_str}")


def install_pystencils_master(session: nox.Session):
    session.install("git+https://i10git.cs.fau.de/pycodegen/pystencils.git@master")


def install_sympy_master(session: nox.Session):
    session.install("--upgrade", "git+https://github.com/sympy/sympy.git@master")


@nox.session(python="3.10", tags=["qa", "code-quality"])
def lint(session: nox.Session):
    """Lint code using flake8"""

    session.install("flake8")
    session.run("flake8", "src/lbmpy")


@nox.session(python="3.10", tags=["qa", "code-quality"])
def typecheck(session: nox.Session):
    """Run MyPy for static type checking"""
    editable_install(session)
    session.install("mypy")
    session.run("mypy", "src/lbmpy")


def run_testsuite(session: nox.Session, coverage: bool = True):
    num_cores = os.cpu_count()

    args = [
        "pytest",
        "-v",
        "-n",
        str(num_cores),
        "-m",
        "not longrun",
        "--html",
        "test-report/index.html",
        "--junitxml=report.xml",
    ]

    if coverage:
        args += [
            "--cov-report=term",
            "--cov=.",
        ]

    session.run(*args)

    if coverage:
        session.run("coverage", "html")
        session.run("coverage", "xml")


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def testsuite_cpu(session: nox.Session):
    install_pystencils_master(session)
    editable_install(session, ["alltrafos", "use_cython", "interactive", "tests"])
    run_testsuite(session, coverage=False)


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
@nox.parametrize("cupy_version", ["12", "13"], ids=["cupy12", "cupy13"])
def testsuite_gpu(session: nox.Session, cupy_version: str | None):
    install_cupy(session, cupy_version, skip_if_no_cuda=True)
    install_pystencils_master(session)
    editable_install(session, ["alltrafos", "use_cython", "interactive", "tests"])
    run_testsuite(session)


@nox.parametrize("cupy_version", [None, "12", "13"], ids=["cpu", "cupy12", "cupy13"])
@nox.session(python="3.10", tags=["test"])
def testsuite_pystencils2(session: nox.Session, cupy_version: str | None):
    if cupy_version is not None:
        install_cupy(session, cupy_version, skip_if_no_cuda=True)

    session.install(
        "git+https://i10git.cs.fau.de/pycodegen/pystencils.git@v2.0-dev"
    )
    editable_install(session, ["alltrafos", "use_cython", "interactive", "tests"])

    run_testsuite(session)


@nox.session
def quicktest(session: nox.Session):
    parser = ArgumentParser()
    parser.add_argument(
        "--sympy-master", action="store_true", help="Use latest SymPy master revision"
    )
    args = parser.parse_args(session.posargs)

    install_pystencils_master(session)
    editable_install(session)

    if args.sympy_master:
        install_sympy_master(session)

    session.run("python", "quicktest.py")
