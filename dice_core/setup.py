import os
import subprocess
import sysconfig

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = os.path.dirname(os.path.abspath(__file__))


def _libcxx_header_args():
    """Workaround for Command Line Tools installs whose toolchain is missing
    the libc++ headers (Apple clang then fails with "'cmath' file not found"):
    point -stdlib++-isystem at the libc++ headers inside the SDK that Python
    was configured against (or the default SDK)."""
    candidates = []
    cflags = sysconfig.get_config_var("CFLAGS") or ""
    parts = cflags.split()
    for i, p in enumerate(parts):
        if p == "-isysroot" and i + 1 < len(parts):
            candidates.append(parts[i + 1])
    try:
        sdk = subprocess.run(
            ["xcrun", "--show-sdk-path"], capture_output=True, text=True
        ).stdout.strip()
        if sdk:
            candidates.append(sdk)
    except OSError:
        pass
    # If the toolchain already provides libc++ headers, do nothing.
    toolchain_cxx = (
        "/Library/Developer/CommandLineTools/usr/include/c++/v1/cmath"
    )
    if os.path.exists(toolchain_cxx):
        return []
    for sdk in candidates:
        hdr = os.path.join(sdk, "usr", "include", "c++", "v1")
        if os.path.exists(os.path.join(hdr, "cmath")):
            return ["-stdlib++-isystem", hdr]
    return []


def _native_arch_args():
    """On x86_64 Linux, target the host CPU's ISA (e.g. AVX2/FMA) so Eigen's
    dense linear algebra in Davidson.cpp (SelfAdjointEigenSolver, the
    subspace matrix ops) and the Hmult matrix-vector products vectorize
    instead of falling back to the generic x86-64 (SSE2-only) baseline.
    Skipped on macOS/ARM, where the toolchain/CPU story is different and
    this file's other workarounds already target Apple clang specifically."""
    import platform
    if platform.system() == "Linux" and platform.machine() == "x86_64":
        return ["-march=native", "-mtune=native"]
    return []

ext_modules = [
    Pybind11Extension(
        "_dice_core",
        sources=[
            "src/global.cpp",
            "src/integral.cpp",
            "src/Determinants.cpp",
            "src/SHCImakeHamiltonian.cpp",
            "src/Davidson.cpp",
            "src/_dice_core.cpp",
        ],
        include_dirs=[
            os.path.join(HERE, "src"),
            os.path.join(HERE, "external", "eigen"),
        ],
        define_macros=[("SERIAL", None), ("NDEBUG", None)],
        cxx_std=17,
        extra_compile_args=["-O3"] + _libcxx_header_args() + _native_arch_args(),
    ),
]

setup(
    name="dice_core",
    version="0.1.0",
    description=(
        "Fixed-basis determinant CI diagonalizer extracted from the Dice "
        "SHCI code (caleb-johnson/Dice), GPLv3"
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
