#!/usr/bin/env python

from setuptools import setup
from distutils.core import setup

from Cython.Build import cythonize

setup(name='fastxml',
      version="0.0.1",
      description='FastXML Extreme Multi-label Classification Algorithm',
      url="https://github.com/refefer/fastxml",
      ext_modules = cythonize([
            "fastxml/splitter.pyx"
      ]),
      packages=['fastxml'],
      scripts=[
      ],
      install_requires=[
        "numpy>=1.8.1",
        "scipy>=0.13.3",
        "scikit-learn>=0.17",
        "Cython>=0.23.4",
      ],
      author='Andrew Stanton')
