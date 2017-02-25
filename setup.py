#!/usr/bin/env python

from setuptools import setup
from distutils.core import setup

from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize

extensions = [
  Extension("fastxml.splitter", ["fastxml/splitter.pyx"], 
      language='c++', extra_compile_args=['-O3', '-std=gnu++11'])
]

setup(name='fastxml',
      version="0.11.0",
      description='FastXML Extreme Multi-label Classification Algorithm',
      url="https://github.com/refefer/fastxml",
      cmdclass = {'build_ext': build_ext},
      ext_modules=extensions,
      packages=['fastxml'],
      scripts=[
          "bin/fxml.py"
      ],
      install_requires=[
        "numpy>=1.8.1",
        "scipy>=0.13.3",
        "scikit-learn>=0.17",
        "Cython>=0.23.4",
      ],
      author='Andrew Stanton')
