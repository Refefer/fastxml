#!/usr/bin/env python
import sys

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

from distutils.extension import Extension

class build_ext(_build_ext):
    """
    This class is necessary because numpy won't be installed at import time.
    """
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

compile_args = ['-O3', '-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.8'] if sys.platform == 'darwin' else ['-O3', '-std=c++11']

extensions = [
  Extension("fastxml.splitter", ["fastxml/splitter.pyx"],
            language='c++',
            extra_compile_args=compile_args),
  Extension("fastxml.inferencer", ["fastxml/inferencer.pyx"],
            language='c++',
            extra_compile_args=compile_args)
]

setup(name='fastxml',
      version="2.0.0",
      description='FastXML Extreme Multi-label Classification Algorithm',
      url="https://github.com/refefer/fastxml",
      author_email="refefer@gmail.com",
      packages=["fastxml"],
      license="LICENSE",
      cmdclass = {'build_ext': build_ext},
      ext_modules=extensions,
      scripts=[
          "bin/fxml.py"
      ],
      install_requires=[
        "numpy>=1.8.1",
        "scipy>=0.13.3",
        "scikit-learn>=0.17",
        "Cython>=0.23.4",
        "future>=0.16.0"
      ],
      classifiers=[
       "License :: OSI Approved :: Apache Software License",
       "Programming Language :: Python :: 2.7",
       "Operating System :: OS Independent"
      ], 
      author='Andrew Stanton')
