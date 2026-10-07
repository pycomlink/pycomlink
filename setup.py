# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:20:35 2014

@author: chwala-c
"""

import os
import re
from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    INSTALL_REQUIRES = [rq for rq in f.read().split("\n") if rq != ""]


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Read the version from pycomlink/__init__.py so there is a single source of
# truth for the package version.
def get_version():
    init_file = os.path.join(os.path.dirname(__file__), "pycomlink", "__init__.py")
    with open(init_file) as f:
        match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', f.read(), re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in pycomlink/__init__.py")
    return match.group(1)


VERSION = get_version()

setup(
    name="pycomlink",
    version=VERSION,
    author="Christian Chwala",
    author_email="christian.chwala@kit.edu",
    description=("Python tools for CML (commercial microwave link) data processing"),
    license="BSD-3-Clause",
    keywords="microwave links precipitation radar cml",
    url="https://github.com/pycomlink/pycomlink",
    download_url=(f"https://github.com/pycomlink/pycomlink/archive/{VERSION}.tar.gz"),
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    # A list of all available classifiers can be found at
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    install_requires=INSTALL_REQUIRES,
)
