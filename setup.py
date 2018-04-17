# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:20:35 2014

@author: chwala-c
"""

import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pycomlink",
    version = "0.2.2",
    author = "Christian Chwala",
    author_email = "christian.chwala@kit.edu",
    description = ("Python tools for MW link data processing"),
    license = "BSD",
    keywords = "microwave links precipitation radar",
    url = "https://github.com/pycomlink/pycomlink",
    download_url = (
        "https://github.com/pycomlink/pycomlink/archive/0.2.2.tar.gz"),
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: BSD License",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6'
    ],
    # A list of all available classifiers can be found at 
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    install_requires=[
        'numpy',
        'scipy',
        'pandas>=0.22',
        'matplotlib',
        'numba',
        'folium',
        'h5py',
        'xarray',
        'shapely',
        'pyproj',
        'tqdm',
        'pykrige',
        'future'],
)
