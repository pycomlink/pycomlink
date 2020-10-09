from __future__ import division
from collections import namedtuple
import numpy as np

from pycomlink.util.maintenance import deprecated


class WetDryError(
    namedtuple(
        "WetDryError",
        [
            "false_wet_rate",
            "missed_wet_rate",
            "matthews_correlation",
            "true_wet_rate",
            "true_dry_rate",
            "N_dry_reference",
            "N_wet_reference",
            "N_true_wet",
            "N_true_dry",
            "N_false_wet",
            "N_missed_wet",
            "N_all_pairs",
            "N_nan_pairs",
            "N_nan_reference_only",
            "N_nan_predicted_only",
        ],
    )
):
    """namedtuple with the following wet-dry performance measures:

    false_wet_rate:
        Rate of cml wet events when reference is dry
    missed_wet_rate:
        Rate of cml dry events when reference is wet
    matthews_correlation:
        Matthews correlation coefficient
    true_wet_rate:
        Rate of cml wet events when the reference is also wet
    true_dry_rate:
        Rate of cml dry events when the reference is also dry
    N_dry_reference:
        Number of dry events in the reference
    N_wet_reference:
        Number of wet events in the reference
    N_true_wet:
        Number of cml wet events when the reference is also wet
    N_true_dry:
        Number of cml dry events when the reference is also dry
    N_false_wet:
        Number of cml wet events when the reference is dry
    N_missed_wet:
        Number of cml dry events when the reference is wet
    N_all_pairs:
        Number of all reference-predicted pairs
    N_nan_pairs:
        Number of reference-predicted pairs with at least one NaN
    N_nan_reference_only:
        Number of NaN values in reference array
    N_nan_predicted_only:
        Number of NaN values in predicted array
    """

    __slots__ = ()


class RainError(
    namedtuple(
        "RainError",
        [
            "pearson_correlation",
            "coefficient_of_variation",
            "root_mean_square_error",
            "mean_absolute_error",
            "R_sum_reference",
            "R_sum_predicted",
            "R_mean_reference",
            "R_mean_predicted",
            "false_wet_rate",
            "missed_wet_rate",
            "false_wet_precipitation_rate",
            "missed_wet_precipitation_rate",
            "rainfall_threshold_wet",
            "N_all_pairs",
            "N_nan_pairs",
            "N_nan_reference_only",
            "N_nan_predicted_only",
        ],
    )
):
    """namedtuple with the following rainfall performance measures:

    pearson_correlation:
        Pearson correlation coefficient
    coefficient_of_variation:
        Coefficient of variation following the definition in[1]
    root_mean_square_error:
        Root mean square error
    mean_absolute_error:
        Mean absolute error
    R_sum_reference:
        Precipitation sum of the reference array (mm)
    R_sum_predicted:
        Precipitation sum of the predicted array (mm)
    R_mean_reference:
        Precipitation mean of the reference array (mm)
    R_mean_predicted:
        Precipitation mean of the predicted array (mm)
    false_wet_rate:
        Rate of cml wet events when reference is dry
    missed_wet_rate:
        Rate of cml dry events when reference is wet
    false_wet_precipitation_rate:
        Mean precipitation rate of false wet events
    missed_wet_precipitation_rate:
        Mean precipitation rate of missed wet events
    rainfall_threshold_wet:
        Threshold separating wet/rain and dry/non-rain periods
    N_all_pairs:
        Number of all reference-predicted pairs
    N_nan_pairs:
        Number of reference-predicted pairs with at least one NaN
    N_nan_reference_only:
        Number of NaN values in the reference array
    N_nan_predicted_only:
        Number of NaN values in predicted array

    References
    -------
    .. [1] Overeem et al. 2013: www.pnas.org/cgi/doi/10.1073/pnas.1217961110

    """

    __slots__ = ()


def calc_wet_dry_performance_metrics(reference, predicted):
    """Calculate performance metrics for a wet-dry classification

    This function calculates metrics and statistics relevant to judge
    the performance of a wet-dry classification. The calculation is based on
    two boolean arrays, where `wet` is True and `dry` is False.

    Parameters
    ----------
    reference : boolean array-like
        Reference values, with `wet` being True
    predicted : boolean array-like
        Predicted values, with `wet` being True

    Returns
    -------
    WetDryError : named tuple

    """

    assert reference.shape == predicted.shape

    # Remove values pairs if either one or both are NaN and calculate nan metrics
    nan_index = np.isnan(reference) | np.isnan(predicted)
    N_nan_pairs = nan_index.sum()
    N_all_pairs = len(reference)
    N_nan_reference_only = np.isnan(reference).sum()
    N_nan_predicted_only = np.isnan(predicted).sum()

    reference = reference[~nan_index]
    predicted = predicted[~nan_index]

    assert reference.shape == predicted.shape

    # force bool type
    reference = reference > 0
    predicted = predicted > 0

    # Calculate N_tp, tn, N_fp, N_fn, N_wet_reference (real positive cases)
    # and N_dry_reference (real negative cases)

    # N_tp is number of true positive wet event (true wet)
    N_tp = ((reference == True) & (predicted == True)).sum()

    # N_tn is number of true negative wet event (true dry)
    N_tn = ((reference == False) & (predicted == False)).sum()

    # N_fp is number of false positive wet event (false wet)
    N_fp = ((reference == False) & (predicted == True)).sum()

    # N_fn is number of false negative wet event  (missed wet)
    N_fn = ((reference == True) & (predicted == False)).sum()

    N_wet_reference = (reference == True).sum()
    N_dry_reference = (reference == False).sum()

    # Then calculate all the metrics
    true_wet_rate = N_tp / N_wet_reference
    true_dry_rate = N_tn / N_dry_reference
    false_wet_rate = N_fp / N_dry_reference
    missed_wet_rate = N_fn / N_wet_reference

    a = np.sqrt(N_tp + N_fp)
    b = np.sqrt(N_tp + N_fn)
    c = np.sqrt(N_tn + N_fp)
    d = np.sqrt(N_tn + N_fn)

    matthews_correlation = ((N_tp * N_tn) - (N_fp * N_fn)) / (a * b * c * d)

    # if predicted has zero/false values only
    # 'inf' would be returned, but 0 is more favorable
    if np.isinf(matthews_correlation):
        matthews_correlation = 0
    if np.nansum(predicted) == 0:
        matthews_correlation = 0

    return WetDryError(
        false_wet_rate=false_wet_rate,
        missed_wet_rate=missed_wet_rate,
        matthews_correlation=matthews_correlation,
        true_wet_rate=true_wet_rate,
        true_dry_rate=true_dry_rate,
        N_dry_reference=N_dry_reference,
        N_wet_reference=N_wet_reference,
        N_true_wet=N_tp,
        N_true_dry=N_tn,
        N_false_wet=N_fp,
        N_missed_wet=N_fn,
        N_all_pairs=N_all_pairs,
        N_nan_pairs=N_nan_pairs,
        N_nan_reference_only=N_nan_reference_only,
        N_nan_predicted_only=N_nan_predicted_only,
    )


def calc_rain_error_performance_metrics(reference, predicted, rainfall_threshold_wet):
    """Calculate performance metrics for rainfall estimation

    This function calculates metrics and statistics relevant to judge
    the performance of rainfall estimation. The calculation is based on
    two arrays with rainfall values, which should contain rain rates or
    rainfall sums. Beware that the units of `R_sum...` und `R_mean...` will
    depend on your input. The calculation does not take any information on
    temporal resolution or aggregation into account!

    Parameters
    ----------
    reference : float array-like
        Rainfall reference
    predicted : float array-like
        Predicted rainfall
    rainfall_threshold_wet : float
        Rainfall threshold for which `reference` and `predicted` are considered
        `wet` if value >= threshold. This threshold only impacts the results
        of the performance metrics which are based on the differentiation
        between `wet` and `dry` periods.

    Returns
    -------
    RainError : named tuple

    References
    -------
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/metrics/regression.py#L184
    https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/metrics/regression.py#L112
    Overeem et al. 2013: www.pnas.org/cgi/doi/10.1073/pnas.1217961110

    """

    assert reference.shape == predicted.shape

    # Remove values pairs if one or both are NaN and calculate nan metrics
    nan_index = np.isnan(reference) | np.isnan(predicted)
    N_nan_pairs = nan_index.sum()
    N_all_pairs = len(reference)
    N_nan_reference_only = np.isnan(reference).sum()
    N_nan_predicted_only = np.isnan(predicted).sum()

    reference = reference[~nan_index]
    predicted = predicted[~nan_index]

    assert reference.shape == predicted.shape

    # calculate performance metrics: pcc, cv, rmse and mae
    pearson_correlation = np.corrcoef(reference, predicted)
    coefficient_of_variation = np.std(predicted - reference) / np.mean(reference)
    root_mean_square_error = np.sqrt(np.mean((predicted - reference) ** 2))
    mean_absolute_error = np.mean(np.abs(predicted - reference))

    # calculate the precipitation sums and mean
    # of reference and predicted time series
    R_sum_reference = reference.sum()
    R_sum_predicted = predicted.sum()
    R_mean_reference = reference.mean()
    R_mean_predicted = predicted.mean()

    # calculate false and missed wet rates and the precipitation at these times
    reference_wet = reference > rainfall_threshold_wet
    reference_dry = ~reference_wet
    predicted_wet = predicted > rainfall_threshold_wet
    predicted_dry = ~predicted_wet

    N_false_wet = (reference_dry & predicted_wet).sum()
    N_dry = reference_dry.sum()
    false_wet_rate = N_false_wet / float(N_dry)

    N_missed_wet = (reference_wet & predicted_dry).sum()
    N_wet = reference_wet.sum()
    missed_wet_rate = N_missed_wet / float(N_wet)

    false_wet_precipitation_rate = predicted[reference_dry & predicted_wet].mean()

    missed_wet_precipitation_rate = reference[reference_wet & predicted_dry].mean()

    return RainError(
        pearson_correlation=pearson_correlation[0, 1],
        coefficient_of_variation=coefficient_of_variation,
        root_mean_square_error=root_mean_square_error,
        mean_absolute_error=mean_absolute_error,
        R_sum_reference=R_sum_reference,
        R_sum_predicted=R_sum_predicted,
        R_mean_reference=R_mean_reference,
        R_mean_predicted=R_mean_predicted,
        false_wet_rate=false_wet_rate,
        missed_wet_rate=missed_wet_rate,
        false_wet_precipitation_rate=false_wet_precipitation_rate,
        missed_wet_precipitation_rate=missed_wet_precipitation_rate,
        rainfall_threshold_wet=rainfall_threshold_wet,
        N_all_pairs=N_all_pairs,
        N_nan_pairs=N_nan_pairs,
        N_nan_reference_only=N_nan_reference_only,
        N_nan_predicted_only=N_nan_predicted_only,
    )


@deprecated("Please use `calc_wet_dry_performance_metrics()`")
def calc_wet_error_rates(df_wet_truth, df_wet):
    N_false_wet = ((df_wet_truth == False) & (df_wet == True)).sum()
    N_dry = (df_wet_truth == False).sum()
    false_wet_rate = N_false_wet / float(N_dry)

    N_missed_wet = ((df_wet_truth == True) & (df_wet == False)).sum()
    N_wet = (df_wet_truth == True).sum()
    missed_wet_rate = N_missed_wet / float(N_wet)

    return WetError(false=false_wet_rate, missed=missed_wet_rate)


WetError = namedtuple("WetError", ["false", "missed"])
