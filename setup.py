from setuptools import setup
import versioneer

with open('README.rst') as f:
    long_description = f.read()

setup(name='pyblis',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      license='BSD',
      description='A Python wrapper for BLIS',
      long_description=long_description,
      packages=['pyblis'],
      include_package_data=True,
      install_requires=["numba"],
      python_requires=">=3.5",
      zip_safe=False)
