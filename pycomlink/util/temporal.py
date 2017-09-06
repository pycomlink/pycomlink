import pandas as pd


def aggregate_df_onto_DatetimeIndex(df, new_index, method, label='right'):
    """
    Aggregate a DataFrame or Series using a given DatetimeIndex

    Parameters
    ----------
    df
    new_index
    method
    label

    Returns
    -------

    df_reindexed

    """

    if label == 'right':
        fill_method = 'bfill'
    elif label == 'left':
        fill_method = 'ffill'
    else:
        raise NotImplementedError('`label` must be "left" or "right"')

    # Make sure we work with a DataFrame and make a copy of it
    df_temp = pd.DataFrame(df).copy()

    # Generate DataFrame with desired DatetimeIndex as data,
    # indexed by DatetimeIndex of original DataFrame
    df_new_t = pd.DataFrame(index=new_index, data={'time': new_index})
    df_new_t = df_new_t.reindex(df_temp.index, method=fill_method)

    # Aggregate data onto new DatetimeIndex
    df_temp['new_time_ix'] = df_new_t.time
    df_reindexed = df_temp.groupby('new_time_ix').agg(method)
    df_reindexed.index.name = df_temp.index.name

    return df_reindexed
