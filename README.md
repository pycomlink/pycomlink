[![Build Status](https://travis-ci.org/pycomlink/pycomlink.svg?branch=master)](https://travis-ci.org/pycomlink/pycomlink)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pycomlink/pycomlink/v0.3alpha)

pycomlink
=========

A python toolbox for deriving rainfall information from commerical microwave link (CML) data.

Installation
------------

`pycomlink` works with Python 2.7, Python 3.6 and Python 3.7. It can be installed via [`conda-forge`](https://conda-forge.org/):

    $ conda install -c conda-forge pycomlink

If you are new to `conda` or if you are unsure, it is recommended to [create a new conda environment, activate it](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands), [add the conda-forge channel](https://conda-forge.org/) and then install.

Installation via `pip` is also possible:

    $ pip install pycomlink

If you install via `pip`, there might be problems with some dependencies, though. Currently the dependency `pykrige` only installs if `scipy`, `numpy` and `matplotlib` have been installed before.

To run the example notebooks you will also need the [Jupyter Notebook](https://jupyter.org/)
and `ipython`, both also available via `conda` or `pip`.

Usage
-----

The following jupyter notebooks showcase some use cases of `pycomlink`

 * [How to do baseline determination](http://nbviewer.jupyter.org/github/pycomlink/pycomlink/blob/master/notebooks/Baseline%20determination.ipynb)
 * [How to do spatial interpolation of CML rainfall](http://nbviewer.jupyter.org/github/pycomlink/pycomlink/blob/master/notebooks/Spatial%20interpolation.ipynb)
 * [How to get started with your CML data from a CSV file](http://nbviewer.jupyter.org/github/pycomlink/pycomlink/blob/master/notebooks/Use%20CML%20data%20from%20CSV%20file.ipynb)

Features
--------
 * Read and write the [common data format `cmlh5` for CML data](https://github.com/cmlh5/cmlh5)
 * Quickly visualize the CML network on a dynamic map
 * Perform all required CML data processing steps to derive rainfall information from raw signal levels:
    * data sanity checks
    * wet/dry classification
    * baseline calculation
    * wet antenna correction
    * transformation from attenuation to rain rate
 * Generate rainfall maps from the data of a CML network
 * Validate you results against gridded rainfall data or rain gauges networks
