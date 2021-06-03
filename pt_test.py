def pt_test(y, yHat):
    import numpy as np
    from scipy.stats import norm
    import collections
    
    if len(y) != len(yHat):
        raise Exception("True data and forecast data have different length")
    
    dy = np.diff(y)
    dyHat = np.diff(yHat)
    
    indicator = lambda x: 1 if x>0 else 0    
    
    Y = np.array([i for i in map(indicator, dy)])
    X = np.array([i for i in map(indicator, dyHat)])
    Z = np.array([i for i in map(lambda x: 1 if x[0]*x[1] > 0 else 0, zip(dy,dyHat))])
    
    n = len(dy)
    Py = sum(Y)/n
    Px = sum(X)/n
    PHat = sum(Z)/len(Z)
    PStar = Py*Px + (1-Py)*(1-Px)
    
    VarPHat = (PStar*(1-PStar))/n
    VarPStar = ((2*Py-1)**2 * Px * (1-Px))/n + ((2*Px-1)**2 * Py * (1-Py))/n + (4 * Px * Py * (1-Py) * (1-Px))/n**2
    
    s = (PHat - PStar) / np.sqrt(VarPHat - VarPStar)
    
    pValue = 1 - norm.cdf(s)
    
    pt_return = collections.namedtuple('PT_test','PT_statistics p_value SuccessR')
    
    rt = pt_return(PT_statistics = s, p_value = pValue, SuccessR = PHat)
    
    return rt
