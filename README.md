pycomlink
=========

A python toolbox for MW link data processing and rain rate estimation

Installation
------------

`pycomlink` works with Python 2.7 and depends on `numpy`, `pandas`, `matplotlib` and `matplotlib`'s `basemap`. You can install it via `pip`, but it is recommended that you install the dependecies first. In particular `basemap` can be challenging to install. We recommend the usage of the [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda/) for all scientific python packages. 

The `pip` command to install `pycomlink` is the following:

    $ pip install pycomlink

Usage
-----

 * IPython notebook showing the [basic workflow](http://nbviewer.ipython.org/urls/bitbucket.org/cchwala/pycomlink/raw/566962d7c0a16c484d56aec7a0a34b84cc68a27d/notebooks/example_workflow.ipynb)
 * IPython notebook on [how to use your MW link CSV data](http://nbviewer.ipython.org/urls/bitbucket.org/cchwala/pycomlink/raw/28f359b359d750434c896d288900844c9b6ef500/notebooks/How%20to%20use%20your%20MW%20link%20data%20from%20a%20CSV%20file.ipynb)

Status
------
The basic functionality is already working. However, `pycomlink` is currently in ongoing development. Hence, there may be changes to the interface in the future.

Features
--------
 * Easily parse your MW link data and use and object oriented approach to do all the processing
 * Two wet/dry classification methods available
 * Different baseline methods available
 * One wet antenna estimation method available
 * 2D plots using IDW (preliminary version) 


