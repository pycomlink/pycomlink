**********************
What's New
**********************

Unreleased
----------


v0.3.5
------

Enhancements
~~~~~~~~~~~~
* Added `bottelneck` as dependency to allow `max_gap` keyword in `xarray.DataArray.interpolate` (by maxmargraf
  in PR #99)
* Added WAA model after Pastorek et al. 2021 (by nblettern via direct commit to master branch)
* Added function and example notebook for blackout gap detection (by maxmargraf in PR #101)
* Refactore and extended grid intersction code, now using sparse matrices (by cchwala in PR #106)

Maintenance
~~~~~~~~~~~~
* Pinned scipy to < 1.9 because of problem in pykrige

Bug fixes
~~~~~~~~~
* Fixed problems in IDW code (by cchwala in PR #105)

v0.3.4
------

Bug fixes
~~~~~~~~~
* Reference files are now included in conda-forge build (PR #97)

Maintenance
~~~~~~~~~~~~
* `tensorflow-gpu` dependency (which seems to be obsolete) was removed from requirements (PR #97)


v0.3.3
------

Enhancements
~~~~~~~~~~~~
* Added xarray-wrapper for WAA Leijnse and updated WAA example notebook (by cchwala
  in PR #82)
* Add CNN-based anomaly detection for CML data (by Glawion in PR#87)
* xarray wrapper now uses `xr.apply_ufunc` to apply processing functions along time
  dimension, instead of looping over the `channel_id` dimension. This should be a lot
  more flexible. (by cchwala in PR #89)

Bug fixes
~~~~~~~~~
* Fixed problem with xarray_wrapper for calc_R_from_A (by cchwala in PR #89)

Maintenance
~~~~~~~~~~~~
* Move CI from Travis to Github Actions (by maxmargraf in PR #85)
* Add readthedocs and zenodo badge to README (by maxmargraaf in PR #85)


v0.3.2
------

* minor fix to include example NetCDF data in source distribution (by cchwala in PR #84)


v0.3.1
------

* small update to how the dependencies are defined
* testing for Python verions 3.7, 3.8 and 3.9


v0.3.0
------

Backward Incompatible Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The old API using `pycomlink.core.Comlink` objects has been removed. All processing
  functions now work with `xarray.DataArrays` or pure `numpy.ndarray`. Most of the
  original functions and notebooks from v0.2.x do not work anymore, but the basic parts
  have already been refactored so that the full processing chain, from raw CML data
  to rainfall fields works in v0.3.0.

Enhancements
~~~~~~~~~~~~

* Added new example notebook for basic processing workflow (by cchwala in PR #77)

* Added new example data (by maxmargraf in PR #75)

* started docs from scratch with working integration to readthedocs (by jpolz in PR #74)

* read data from cmlh5 files to `xarray.Dataset` (by maxmargraf in PR #68)

* Added functions to perform wet-dry classification with trained CNN (by jpolz in PR #67)

* applied black formatting to codebase (by nblettner in PR #66)

* make repo runnable via mybinder (by jpolz in PR #64)


v0.2.4
------

* Added WAA calculation and test for method proposed by Leijnse et al 2008

* Added function to calculate WAA directly from A_obs for Leijnse et al 2008
  method.

* Added WAA example notebook

* Added function to derive attenuation value `A_min_max` from min/max CML
  measurements (these measurements periodically provide the min and max
  value over a defined time period, typically 15 minutes).
  (by DanSereb in PR #37 and #45)

* Added function to derive rain rate `R` from `A_min_max`
  (by DanSereb in PR #37 and #45)

* Added example notebook with simple comparison of processing of
  "instantaneous" and "min-max" CML data  (by DanSereb in PR #37 and #45)


v0.2.3
------

Bug fixes
~~~~~~~~~

* Added missing kwarg for polarization in `calc_A` in `Processor`. Before,
  `calc_A` always used the default polarization for the A-R relation which
  leads to rain rate overestimation!

* Changed reference values in test for Ordinary Kriging interpolator, because
  `pykrige v1.4.0` seems to produce slightly different results than `v1.3.1`

v0.2.2
------

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
