What's New
==========

v0.2.0 (unreleased)
-------------------

Backward Incompatible Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-Complete rewrite of interpolator classes. The old interpolator class
 `spatial.interpol.Interpolator()` is depreciated. New interpolator base classes
 for IDW and Kriging have been added together with a convenience inteprolator
 for CML data. Usage is showcased in a new example notebook.

-Some old functionality has moved to separate files.
  * resampling to a given `DatetimeIndex` is now availabel in `util.temporal`
    and will be removed from `validatoin.validator.Validation()` class soon.
  * calculation of wet-dry error is now in module `validation.stats`
  * calculation of spatial coverage with CMLs was moved to function
    `spatial.coverage.calc_coverage_mask()`.

-Added new error metric for performance evaluation of wet-dry classification.
 The old one in `validation.validator.calc_wet_dry_error()` is depreciated now
 since its `dry_error` is not optimal.

Enhancements
~~~~~~~~~~~~

-Read and write to and from multiple cmlh5 files (#12)

-Improved `NaN` handling in `wet` indicator for baseline determination

-Added example notebook for baseline determination

-Added data set of 75 CMLs (with fake locations)

-Added example notebook to show usage of new interpolator classes

-Added decorator to mark depreciated code

Bug fixes
~~~~~~~~~

-`setup.py` now reads all packages subdirectories correctly

-Force integers for shape in `nans` helper function in `stft` module

-Always use first value of `dry_stop` timestamp list in `stft` module.
 The old code did not work anyway for a list with length = 1 and would
 have failed if `dry_stop` would have been a scalar value. Now we
 assume that we always get a list of values (which should be true for
 `mlab.find`.


v0.1.1
------

No info for older version...