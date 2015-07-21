from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([
        Extension("numerics", ["numerics.pyx"],
                  include_dirs=[numpy.get_include()]),
    ]),
    include_dirs=[numpy.get_include()],
)
