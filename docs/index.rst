.. pycomlink documentation master file, created by
   sphinx-quickstart on Wed Jul  1 16:21:41 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation of pycomlink
==========================

A python toolbox for microwave link data processing and rain rate estimation

**Features**
 * Easily parse your MW link data and use and object oriented approach to do all the processing
 * Two wet/dry classification methods available
 * Different baseline methods available
 * One wet antenna estimation method available
 * 2D plots using IDW or Kriging interpolation
 
**Installation**

    >> pip install pycomlink

**Usage**
 * IPython notebook showing the `basic workflow <http://nbviewer.ipython.org/urls/bitbucket.org/cchwala/pycomlink/raw/566962d7c0a16c484d56aec7a0a34b84cc68a27d/notebooks/example_workflow.ipynb>`_.
 * IPython notebook on `how to use your MW link CSV data <http://nbviewer.ipython.org/urls/bitbucket.org/cchwala/pycomlink/raw/28f359b359d750434c896d288900844c9b6ef500/notebooks/How%20to%20use%20your%20MW%20link%20data%20from%20a%20CSV%20file.ipynb>`_.
 
.. toctree::
   :maxdepth: 2
   
Main Classes:
-------------
.. toctree::

   pycomlink.Comlink
   pycomlink.ComlinkSet
   
Submodules:
-----------
.. toctree::

   pycomlink
  

Indices and tables:
-------------------
* :ref:`genindex`
* :ref:`modindex`
