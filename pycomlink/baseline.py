

################################################
# Functions for setting the RSL baseline level #
################################################

def baseline_constant(rsl, wet):
    import numpy as np
    baseline = np.zeros(np.shape(rsl))
    baseline[0] = rsl[0]
    for i in range(1,len(rsl)):
        if wet[i]:
            baseline[i] = baseline[i-1]
        else:
            baseline[i] = rsl[i]
    return baseline
            
def baseline_linear(rsl, wet):
    import numpy as np
    baseline = np.zeros(np.shape(rsl))
    baseline[0] = rsl[0]
    last_dry_i = 0
    last_i_is_wet = False
    for i in range(1,len(rsl)):
        is_wet = wet[i]
        # Check for NaN values. If NaN, then continue with
        # the last wet/dry state
        if np.isnan(is_wet):
            is_wet = last_i_is_wet
        # at the begining of a wet period
        if is_wet and not last_i_is_wet:
            last_i_is_wet = True
        # within a wet period
        if is_wet and last_i_is_wet:
            last_i_is_wet = True
        # at the end of a wet period, do the baseline interpolation
        elif last_i_is_wet and not is_wet:
            #!! Only works correctly with 'i+1'. With 'i' the first dry
            #!! baseline value is kept at 0. No clue why we need the '+1'
            baseline[last_dry_i:i+1] = np.linspace(rsl[last_dry_i], 
                                                 rsl[i],
                                                 i-last_dry_i+1)
            last_i_is_wet = False
            last_dry_i = i
        # within a dry period
        elif not last_i_is_wet and not is_wet:
            baseline[i] = rsl[i]
            last_i_is_wet = False             
            last_dry_i = i
        else:
            print 'This should be impossible'
            raise
    return baseline