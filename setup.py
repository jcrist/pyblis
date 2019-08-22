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
GENERATE_SCRIPT = os.path.join(LIB_SRC_DIR, "generate.py")
LIB_BUILD_OUTPUT = os.path.join(LIB_BUILD_DIR, "libpyblis.%s" % EXT)
LIB_TGT_DIR = os.path.join(ROOT_DIR, "pyblis")
LIB_TGT = os.path.join(LIB_TGT_DIR, "_lib.%s" % EXT)
PY_SOURCE_TGT = os.path.join(LIB_TGT_DIR, "_lib.py")
PY_SOURCE_TEMPLATE = os.path.join(LIB_TGT_DIR, "_lib.py.template")


@contextlib.contextmanager
def changed_dir(dirname):
    oldcwd = os.getcwd()
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(oldcwd)


def _ensure_jinja2(cmd):
    try:
        import jinja2  # noqa
    except ImportError:
        cmd.warn('Building pyblis requires jinja2, please install and try again')
        sys.exit(1)


def _ensure_lib(cmd):
    if not os.path.exists(LIB_TGT):
        cmd.run_command("build_ext")
    if not os.path.exists(PY_SOURCE_TGT):
        cmd.run_command("gen_py_source")


class build_ext(Command):
    description = "build blis wrapper ext"

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
        _ensure_jinja2(self)
        cmake_options = [
            "-DPYBLIS_BUILD_BLIS=" + ("on" if self.bundle_blis else "off"),
            "-DPYBLIS_BUNDLE_BLIS=" + ("on" if self.bundle_blis else "off")
        ]
        os.makedirs(LIB_BUILD_DIR, exist_ok=True)
        with changed_dir(LIB_BUILD_DIR):
            self.spawn(["cmake"] + cmake_options + [LIB_SRC_DIR])
            self.spawn(["cmake", "--build", "."])
            self.copy_file(LIB_BUILD_OUTPUT, LIB_TGT)


class gen_py_source(Command):
    description = "generate python lib wrapper source"

    user_options = []

    def initialize_options(self):
        pass

    finalize_options = initialize_options

    def run(self):
        _ensure_jinja2(self)
        self.spawn([sys.executable, GENERATE_SCRIPT, PY_SOURCE_TEMPLATE, PY_SOURCE_TGT])


class build(_build):
    def run(self):
        _ensure_lib(self)
        _build.run(self)


class install(_install):
    def run(self):
        _ensure_lib(self)
        _install.run(self)


class develop(_develop):
    def run(self):
        if not self.uninstall:
            _ensure_lib(self)
        _develop.run(self)


class clean(_clean):
    def run(self):
        if self.all:
            for f in [LIB_TGT, PY_SOURCE_TGT]:
                if os.path.exists(f):
                    os.unlink(f)
        _clean.run(self)


# Due to quirks in setuptools/distutils dependency ordering, to get the lib to
# build automatically in most cases, we need to check in multiple locations.
# This is unfortunate, but seems necessary.
cmdclass = versioneer.get_cmdclass()
cmdclass.update(
    {
        "build_ext": build_ext,             # directly build the ext source
        "gen_py_source": gen_py_source,     # directly build the python source
        "build": build,                     # bdist_wheel or pip install .
        "install": install,                 # python setup.py install
        "develop": develop,                 # python setup.py develop
        "clean": clean,                     # extra cleanup
    }
)

extras_require = {
    "numba": ["numba"],
}

install_requires = ["numpy"]

entry_points = {
    "numba_extensions": [
        "init = pyblis:_init_numba",
    ],
}

setup(name='pyblis',
      version=versioneer.get_version(),
      cmdclass=cmdclass,
      license='BSD',
      description='A Python wrapper for BLIS',
      long_description=open('README.rst').read(),
      packages=['pyblis'],
      include_package_data=True,
      python_requires=">=3.5",
      extras_require=extras_require,
      install_requires=install_requires,
      entry_points=entry_points,
      zip_safe=False)
