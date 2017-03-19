pycomlink
=========

A python toolbox for deriving rainfall information from commerical microwave link (CML) data.

Installation
------------

`pycomlink` works with Python 2.7 and can be installed via `pip`. However, since one of its dependencies, `numba` is easiest to install via the [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda/), we recommend to install Anaconda Python first and then do

    $ conda install numba
    $ pip install pycomlink

To run the example notebooks you will also need the [Jupyter Notebook](https://jupyter.org/) and `ipython`, both also available via `conda` or `pip`.

Usage
-----

 * Jupyter notebook on [how to get started with CML data from a CSV file](http://nbviewer.jupyter.org/github/pycomlink/pycomlink/blob/master/notebooks/Use%20CML%20data%20from%20CSV%20file.ipynb)
 * More examples to come...

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
