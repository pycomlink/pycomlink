[![Build Status](https://travis-ci.org/pycomlink/pycomlink.svg?branch=master)](https://travis-ci.org/pycomlink/pycomlink)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pycomlink/pycomlink/master)

pycomlink
=========

A python toolbox for deriving rainfall information from commercial microwave link (CML) data.

Installation
------------

`pycomlink` works with Python 3.6 and newer. It might still work with Python 2.7, but this is not tested. It can be installed via [`conda-forge`](https://conda-forge.org/):

    $ conda install -c conda-forge pycomlink

If you are new to `conda` or if you are unsure, it is recommended to [create a new conda environment, activate it](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands), [add the conda-forge channel](https://conda-forge.org/) and then install.

Installation via `pip` is also possible:

    $ pip install pycomlink

If you install via `pip`, there might be problems with some dependencies, though. E.g. the dependency `pykrige` may only install if `scipy`, `numpy` and `matplotlib` have been installed before.

To run the example notebooks you will also need the [Jupyter Notebook](https://jupyter.org/)
and `ipython`, both also available via `conda` or `pip`.

If you want to clone the repository for developing purposes follow these steps (installation of Jupyter Notebook included):

    $ cd WORKING_DIRECTORY
    $ git clone https://github.com/pycomlink/pycomlink.git
    $ conda env create --name ENV_NAME -file=environment_dev.yml
    $ conda activate ENV_NAME
    $ pip install -e WORKING_DIRECTORY/pycomlink

Usage
-----

The following jupyter notebooks showcase some use cases of `pycomlink`

 * [Basic example CML processing workflow](http://nbviewer.jupyter.org/github/pycomlink/pycomlink/blob/master/notebooks/Basic%20CML%20processing%20workflow.ipynb)
 * more to come... (see some [notebooks with old outdated pycomlink API](https://github.com/pycomlink/pycomlink/tree/master/notebooks/outdated_notebooks))

Features
--------

 * Perform all required CML data processing steps to derive rainfall information from raw signal levels:
    * data sanity checks
    * wet/dry classification
    * baseline calculation
    * wet antenna correction
    * transformation from attenuation to rain rate
 * Generate rainfall maps from the data of a CML network
 * Validate you results against gridded rainfall data or rain gauges networks
