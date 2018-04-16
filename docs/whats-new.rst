What's New
==========

v0.2.2 (not released)
---------------------

Enhancements
~~~~~~~~~~~~

* Codebase is Python 3 now, keeping backwards compatibility to Python 2.7
  via using the `future` module.

* min-max CML data can now be written to and read from cmlh5. Standard column
  names are `tx_min`, `tx_max`, `rx_min` and `rx_max`. When reading from cmlh5
  without specifying dedicated column names, the function tries out the
  standard column names for min-max and instantaneous. If it does not find any
  match it will print an error message.

* Added example file with min-max data for 75 CMLs. This dataset is derived
  from the existing example dataset of 75 CMLs with instantaneous measurements.

* Added example notebook comparing min-max and instantaneous CML data

* Added TravisCI and Codecov and increased the test coverage a little

* Extended functionality for `append_data`. A maximum length or maximum
  allowed age for the data can be specified

* More options for interpolation. Added option to pass `max_distance`
  for IDW and Added option for resampling in `Interpolator`
  (instead of just doing hourly means of variable `R`)

* Interpolated fields are now always transformed into an `xarray.Dataset`.
  The `Dataset` is also stored as attribute if the `Interpolator` object

* Improved grid intersection calculation in validator

Bug fixes
~~~~~~~~~

* `t_start` and `t_stop` have not been taken into account
  in the main interpolation loop

* Fix: Catching `LinAlgError` in Kriging interpolation


v0.2.1
------

Minor update

* removing geopandas dependecy
* update MANIFEST.in to include notebooks and example data in pypi releases


v0.2.0
------

Backward Incompatible Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Complete rewrite of interpolator classes. The old interpolator class
  `spatial.interpol.Interpolator()` is depreciated. New interpolator base classes
  for IDW and Kriging have been added together with a convenience inteprolator
  for CML data. Usage is showcased in a new example notebook.

* Some old functionality has moved to separate files.
    * resampling to a given `DatetimeIndex` is now availabel in `util.temporal`
      and will be removed from `validatoin.validator.Validation()` class soon.
    * calculation of wet-dry error is now in module `validation.stats`
    * calculation of spatial coverage with CMLs was moved to function
      `spatial.coverage.calc_coverage_mask()`.
    * error metric for performance evaluation of wet-dry classification is now
      in `validation.stats`. Errors are now returned with meaningful names as
      namedtuples. `validation.validator.calc_wet_dry_error()` is depreciated now.

Enhancements
~~~~~~~~~~~~

* Read and write to and from multiple cmlh5 files (#12)

* Improved `NaN` handling in `wet` indicator for baseline determination

* Speed up of KDtreeIDW using numba and by reusing
  previously calculated variables

* Added example notebook for baseline determination

* Added data set of 75 CMLs (with fake locations)

* Added example notebook to show usage of new interpolator classes

* Added decorator to mark depreciated code

Bug fixes
~~~~~~~~~

* `setup.py` now reads all packages subdirectories correctly

* Force integers for shape in `nans` helper function in `stft` module

* Always use first value of `dry_stop` timestamp list in `stft` module.
  The old code did not work anyway for a list with length = 1 and would
  have failed if `dry_stop` would have been a scalar value. Now we
  assume that we always get a list of values (which should be true for
  `mlab.find`.


v0.1.1
------

No info for older version...