from collections import namedtuple


WetError = namedtuple('WetError', ['false', 'missed'])


def calc_wet_error_rates(df_wet_truth, df_wet):
    N_false_wet = ((df_wet_truth == False) & (df_wet == True)).sum()
    N_wet = (df_wet_truth == True).sum()
    false_wet_rate = N_false_wet / float(N_wet)

    N_missed_wet = ((df_wet_truth == True) & (df_wet == False)).sum()
    missed_wet_rate = N_missed_wet / float(N_wet)

    return WetError(false=false_wet_rate, missed=missed_wet_rate)