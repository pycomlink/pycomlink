import pandas as pd


def aggregate_df_onto_DatetimeIndex(
    df, new_index, method, label="right", new_index_tz="utc"
):
    """
    Aggregate a DataFrame or Series using a given DatetimeIndex

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that should be reindexed
    new_index : pandas.DatetimeIndex
        The time stamp index on which `df` should be aggregated
    method : numpy function
        The function to be used for aggregation via
        `DataFrame.groupby('new_time_ix').agg(method)`
    label : str {'right', 'left'}, optional
        Which side of the aggregated period to take the label for the new
        index from
    new_index_tz : str, optional
        Defaults to 'utc'. Note that if `new_index` already has time zone
        information, this kwarg is ignored

    Returns
    -------

    df_reindexed : pandas.DataFrame

    """

    if label == "right":
        fill_method = "bfill"
    elif label == "left":
        fill_method = "ffill"
    else:
        raise NotImplementedError('`label` must be "left" or "right"')

    # Make sure we work with a DataFrame and make a copy of it
    df_temp = pd.DataFrame(df).copy()

    # Generate DataFrame with desired DatetimeIndex as data,
    # which will later be reindexed by DatetimeIndex of original DataFrame
    df_new_t = pd.DataFrame(index=new_index, data={"time": new_index})

    # Update time zone info if there is none
    if not df_new_t.index.tzinfo:
        df_new_t.index = df_new_t.index.tz_localize(new_index_tz)

    # Crop both time series to make them cover the same period.
    # This is to avoid the ffill or bfill to run outside of the
    # range of the new index, which produces wrong result for the
    # end point of the time series in the aggregated result
    t_start = max(df_temp.index.min(), df_new_t.index.min())
    t_stop = min(df_temp.index.max(), df_new_t.index.max())
    df_new_t = df_new_t.loc[t_start:t_stop]
    df_temp = df_temp.loc[t_start:t_stop]

    # Reindex to get the forward filled or backwar filled time stamp of the
    # new index which can be used for aggregation in the next step
    df_new_t = df_new_t.reindex(df_temp.index, method=fill_method)

    # Aggregate data onto new DatetimeIndex
    df_temp["new_time_ix"] = df_new_t.time
    df_reindexed = df_temp.groupby("new_time_ix").agg(method)
    # Update name and timezone of new index
    df_reindexed.index.name = df_temp.index.name
    if not df_reindexed.index.tzinfo:
        df_reindexed.index = df_reindexed.index.tz_localize("UTC").tz_convert(
            df_temp.index.tzinfo
        )

    return df_reindexed
