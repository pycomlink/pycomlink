[![Build Status](https://travis-ci.org/pycomlink/pycomlink.svg?branch=master)](https://travis-ci.org/pycomlink/pycomlink)

pycomlink
=========

A python toolbox for deriving rainfall information from commerical microwave link (CML) data.

Installation
------------

`pycomlink` works with Python 2.7 and Python 3.6 and can be installed via `pip`.

    $ pip install pycomlink

However, for using scientific Python packages it is in general recommended to 
install the [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda/) and use
its package manager `conda` for managing all Python packages. `pycomlink` is, however,
not yet installable via the Anaconda community package channel [conda-forge](https://conda-forge.org/).
Hence, it is recommended to install all `pycomlink` dependencies (listed in `requirements.txt`) 
via `conda` and then use `pip` to install `pycomlink`. 

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
