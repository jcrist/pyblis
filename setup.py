import contextlib
import os
import sys
from distutils.command.build import build as _build
from distutils.command.clean import clean as _clean

from setuptools import setup, Command
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install

import versioneer

if sys.platform == "darwin":
    EXT = "dylib"
elif sys.platform == "win32":
    EXT = "dll"
else:
    EXT = "so"

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.relpath(__file__)))
LIB_SRC_DIR = os.path.join(ROOT_DIR, "lib")
LIB_BUILD_DIR = os.path.join(LIB_SRC_DIR, "build")
LIB_BUILD_OUTPUT = os.path.join(LIB_BUILD_DIR, "libpyblis.%s" % EXT)
LIB_TGT_DIR = os.path.join(ROOT_DIR, "pyblis")
LIB_TGT = os.path.join(LIB_TGT_DIR, "_lib.%s" % EXT)


@contextlib.contextmanager
def changed_dir(dirname):
    oldcwd = os.getcwd()
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(oldcwd)


class build_lib(Command):
    description = "build blis wrapper lib"

    user_options = [
        ("bundle-blis", None, "bundle BLIS with the library"),
        ("build-blis", None, "build BLIS rather than using an installed version")
    ]

    def initialize_options(self):
        self.bundle_blis = False
        self.build_blis = False

    def finalize_options(self):
        pass

    def run(self):
        cmake_options = [
            "-DPYBLIS_BUILD_BLIS=" + ("on" if self.bundle_blis else "off"),
            "-DPYBLIS_BUNDLE_BLIS=" + ("on" if self.bundle_blis else "off")
        ]
        os.makedirs(LIB_BUILD_DIR, exist_ok=True)
        with changed_dir(LIB_BUILD_DIR):
            self.spawn(["cmake"] + cmake_options + [LIB_SRC_DIR])
            self.spawn(["cmake", "--build", "."])
            self.copy_file(LIB_BUILD_OUTPUT, LIB_TGT)


def _ensure_lib(command):
    if not getattr(command, "no_lib", False) and not os.path.exists(LIB_TGT):
        command.run_command("build_lib")


class build(_build):
    def run(self):
        _ensure_lib(self)
        _build.run(self)


class install(_install):
    def run(self):
        _ensure_lib(self)
        _install.run(self)


class develop(_develop):
    user_options = list(_develop.user_options)
    user_options.append(("no-go", None, "Don't build the go source"))

    def initialize_options(self):
        self.no_go = False
        _develop.initialize_options(self)

    def run(self):
        if not self.uninstall:
            _ensure_lib(self)
        _develop.run(self)


class clean(_clean):
    def run(self):
        if self.all:
            for f in [LIB_TGT]:
                if os.path.exists(f):
                    os.unlink(f)
        _clean.run(self)


# Due to quirks in setuptools/distutils dependency ordering, to get the lib to
# build automatically in most cases, we need to check in multiple locations.
# This is unfortunate, but seems necessary.
cmdclass = versioneer.get_cmdclass()
cmdclass.update(
    {
        "build_lib": build_lib,     # directly build the lib source
        "build": build,             # bdist_wheel or pip install .
        "install": install,         # python setup.py install
        "develop": develop,         # python setup.py develop
        "clean": clean,             # extra cleanup
    }
)


setup(name='pyblis',
      version=versioneer.get_version(),
      cmdclass=cmdclass,
      license='BSD',
      description='A Python wrapper for BLIS',
      long_description=open('README.rst').read(),
      packages=['pyblis'],
      include_package_data=True,
      setup_requires=["jinja2"],
      install_requires=["numba"],
      python_requires=">=3.5",
      zip_safe=False)
