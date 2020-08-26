import pandas as pd
import numpy as np
import scipy
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from scipy.stats import levene, jarque_bera

# =============================================================================
# # Import the EONIA time series
# =============================================================================
df = pd.read_csv("C:/Users/s157148/Documents/Research/Data/EONIA_rate.csv", sep=";")
df_ester = pd.read_csv("C:/Users/s157148/Documents/Research/Data/ESTER_rate.csv", sep=";")
df_ester = df_ester.iloc[:115, :]
df_pre_ester = pd.read_csv("C:/Users/s157148/Documents/Research/Data/PRE_ESTER.csv", sep=";")

# =============================================================================
# Calculate the strength of trend and seasonality
# =============================================================================

# Calculate the strength of trend and seasonality
def strength(data):
    '''Calculate the strength of trend and seasonality.'''
    
    # Fit the STL decomposition
    res = STL(data, period = 5, seasonal = 5, robust = True).fit()
    
    strength_trend = np.maximum(0.0, 1 - np.var(res.resid) / np.var(res.trend + res.resid))
    strength_season = np.maximum(0.0, 1 - np.var(res.resid) / np.var(res.seasonal + res.resid))
    
    return [strength_trend, strength_season]

# =============================================================================
# Calculate the spikiness
# =============================================================================

def spikiness(data):
    '''Calculate the spikiness.'''
    
    # Fit the STL decomposition
    res = STL(data, period = 5, seasonal = 5, robust = True).fit()
    
    # Fit the Leave-One-Out variances
    loo_vars = [ np.var(res.resid[np.arange(len(res.resid)) != i]) for i in range(0, len(res.resid)) ]
    
    # Compute spikiness as the variances of the Leave-One-Out variances
    return np.var(loo_vars)

# =============================================================================
# Calculate the Hurst exponent
# =============================================================================

def hurst(data):
    '''Calculate the Hurst exponent.'''
    # Define the lags over which we check the autocorrelation
    lags = range(2, 100)
    
    # calculate standard deviation of differenced series using various lags
    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
    
    # Fit a linear regression through the log-log plot
    m = np.polyfit(x=np.log(lags), y=np.log(tau), deg=1)
    return(m[0]*2.0)

# =============================================================================
# Calculate the characteristics of the interest rates
# =============================================================================

def descriptives_over_time():
    '''Calculate the characteristics of the interest rate returns over time.'''
    
    data = np.ravel(pd.read_csv("C:/Users/s157148/Documents/Research/Data/PRE_ESTER.csv", sep=";").PRE_ESTER)
    descriptives = np.zeros((len(data)-150, 8))
    for i in range(len(data)-150):
        temp = data[i:(i+150)]
        descriptives[i,0] = np.var(temp)
        descriptives[i,1] = scipy.stats.skew(temp)
        descriptives[i,2] = scipy.stats.kurtosis(temp)
        descriptives[i,3] = hurst(temp)
        descriptives[i,4] = adfuller(temp)[1]
        strength_measures = strength(temp)
        descriptives[i,5] = strength_measures[0]
        descriptives[i,6] = strength_measures[1]
        descriptives[i,7] = spikiness(temp)
    return descriptives

def descriptives(data):
    '''Calculate the descriptives of the interest rate returns.'''
    
    count = len(data)
    variance = np.var(data)
    skewness = scipy.stats.skew(data)
    kurtosis = scipy.stats.kurtosis(data)
    hurst_exp = hurst(data)
    p_val_adf = adfuller(data)[1]

    strength_measures = strength(data)
    strength_trend = strength_measures[0]
    strength_season = strength_measures[1]
    
    spike = spikiness(data)

    print('\n Count:', count,
          '\n Variance:', np.round(variance, 5),
          '\n Skewness:', np.round(skewness, 5), 
          '\n Kurtosis:', np.round(kurtosis, 5),
          '\n Hurst exponents:', np.round(hurst_exp, 5),
          '\n p-value ADF', np.round(p_val_adf, 5),
          '\n Strength trend', np.round(strength_trend, 5),
          '\n Strength season', np.round(strength_season, 5), 
          '\n Spikiness', spike)
    return 'Done'

# =============================================================================
# Calculate the interest rate returns and provide the descriptives
# =============================================================================

# Calculate EONIA + 8.5 bps
df_eonia_ester = df[4662:5311]
df_eonia_ester.columns = ['Date', 'ECB_mapping']
df_eonia_ester.ECB_mapping = df_eonia_ester.ECB_mapping.values - 0.085 
df_eonia_ester.ECB_mapping[1:] = df_eonia_ester.ECB_mapping.pct_change()[1:]
df_eonia_ester = df_eonia_ester.iloc[1:,] 

# # Calculate the daily differences
# df.EONIA[1:] = np.diff(df.EONIA)
df.EONIA[1:] = df.EONIA.pct_change()[1:]
df = df.iloc[1:, :]


# df_pre_ester.PRE_ESTER[1:] = np.diff(df_pre_ester.PRE_ESTER)
df_pre_ester.PRE_ESTER[1:] = df_pre_ester.PRE_ESTER.pct_change()[1:]
df_pre_ester = df_pre_ester.iloc[1:, :] 

# df_ester.ESTER[1:] = np.diff(df_ester.ESTER)
df_ester.ESTER[1:] = df_ester.ESTER.pct_change()[1:]
df_ester = df_ester.iloc[1:, :]

descriptives(df.EONIA.values)
descriptives(df.EONIA.values[4662:5310])
descriptives(df_eonia_ester.ECB_mapping.values)
descriptives(df_pre_ester.PRE_ESTER.values)    
descriptives(df_ester.ESTER.values)

#test = descriptives_over_time(df_pre_ester.PRE_ESTER.values)

# =============================================================================
# Test equality of variance, skewness and kurtosis
# =============================================================================

non_normal_1 = jarque_bera(df_eonia_ester.ECB_mapping.values)[1] < 0.05
non_normal_2 = jarque_bera(df_pre_ester.PRE_ESTER.values)[1] < 0.05
equal_variances = levene(df_eonia_ester.ECB_mapping, df_pre_ester.PRE_ESTER)[1] < 0.05

print('The ECB mapping is non normal: ' + str(non_normal_1))
print('The ESTER returns are non normal: ' + str(non_normal_2))
print('The variances are equal: ' + str(equal_variances))
