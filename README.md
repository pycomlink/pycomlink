[![CI](https://github.com/pycomlink/pycomlink/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/pycomlink/pycomlink/actions/workflows/main.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pycomlink/pycomlink/master)
[![Documentation Status](https://readthedocs.org/projects/pycomlink/badge/?version=latest)](https://pycomlink.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4810169.svg)](https://doi.org/10.5281/zenodo.4810169)

Anaconda Version [![Anaconda Version](https://anaconda.org/conda-forge/pycomlink/badges/version.svg)](https://anaconda.org/conda-forge/pycomlink) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/pycomlink/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/pycomlink)

pycomlink
=========

A python toolbox for deriving rainfall information from commercial microwave link (CML) data.

Installation
------------

`pycomlink` is tested with Python 3.10, 3.11, 3.12, and 3.13. Many things might work with older version, but there is no support for this.

It can be installed via [`conda-forge`](https://conda-forge.org/):

    $ conda install -c conda-forge pycomlink

If you are new to `conda` or if you are unsure, it is recommended to [create a new conda environment, activate it](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands), [add the conda-forge channel](https://conda-forge.org/) and then install.

Installation via `pip` is also possible:

    $ pip install pycomlink

At the time of writing, with `pycomlink v0.4.0` which dropped `tensorflow` as dependency, `pip` install works fine. But, if we add new dependencies in the future, we might again run into issues with `pip` install.

To run the example notebooks you will also need the [Jupyter Notebook](https://jupyter.org/)
and `ipython`, both also available via `conda` or `pip`.

If you want to clone the repository for developing purposes follow these steps (installation of Jupyter Notebook included):

    $ git clone https://github.com/pycomlink/pycomlink.git
    $ cd pycomlink
    $ conda env create -f environment_dev.yml
    $ conda activate pycomlink-dev
    $ cd ..
    $ pip install -e pycomlink

Usage
-----

The following jupyter notebooks showcase some use cases of `pycomlink`

 * [Basic example CML processing workflow](http://nbviewer.jupyter.org/github/pycomlink/pycomlink/blob/master/notebooks/Basic%20CML%20processing%20workflow.ipynb)
 * [Compare interpolation methods](https://nbviewer.org/github/pycomlink/pycomlink/blob/master/notebooks/Compare%20interpolation%20methods.ipynb)
 * [Get radar data along CML paths](https://nbviewer.org/github/pycomlink/pycomlink/blob/master/notebooks/Get%20radar%20rainfall%20along%20CML%20paths.ipynb)
 * [Nearby-link approach for rain event detection from RAINLINK](https://nbviewer.org/github/pycomlink/pycomlink/blob/master/notebooks/Nearby%20link%20approach%20processing%20example.ipynb)
 * [Compare different WAA methods](https://nbviewer.org/github/pycomlink/pycomlink/blob/master/notebooks/Wet%20antenna%20attenuation.ipynb)
 * [Detect data gaps stemming from heavy rainfall events that cause a loss of connection along a CML](https://nbviewer.org/github/pycomlink/pycomlink/blob/master/notebooks/Blackout%20gap%20detection%20examples.ipynb)

Note that the links point to static versions of the example notebooks. You can run all these notebook online via mybinder if you click on the "launch binder" buttom at the top.

Features
--------

 * Perform all required CML data processing steps to derive rainfall information from raw signal levels:
    * data sanity checks
    * ~~anomaly detection~~ (removed because using outdated `tensorflow` code)
    * wet/dry classification
    * baseline calculation
    * wet antenna correction
    * transformation from attenuation to rain rate
 * Generate rainfall maps from the data of a CML network
 * Validate you results against gridded rainfall data or rain gauges networks
 
Documentation
-------------
The documentation is hosted by readthedocs.org: [https://pycomlink.readthedocs.io/en/latest/](https://pycomlink.readthedocs.io/en/latest/)
