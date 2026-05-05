from pathlib import Path
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
os.chdir(REPO_ROOT)

ext_modules = [
    Pybind11Extension(
        "model.gsamd._gsamd_core",
        [str(ROOT / "cpp" / "gsamd_core.cpp")],
        cxx_std=17,
    ),
]


setup(
    name="gsamd",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
