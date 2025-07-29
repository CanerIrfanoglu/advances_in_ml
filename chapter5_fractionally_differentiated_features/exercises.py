# 🎯 What it does in simple terms:
# Instead of subtracting only the previous value (like in P_t - P_{t-1}),

# It subtracts a weighted sum of previous values.

# For example:

# new_value_t = P_t - 0.9 * P_{t-1} - 0.8 * P_{t-2} - 0.7 * P_{t-3} ...
# The weights decay over time.
# Smaller d → slower decay → more memory kept.
# Larger d → faster decay → less memory, more like regular differencing.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Set seed for reproducibility
np.random.seed(42)

# 5.1 Generate a time series from an IID Gaussian process (mean=0, std=1) (memoryless, stationary)
n = 500
iid_series = np.random.normal(loc=0, scale=1, size=n)

# (a) ADF test on the original series (stationary)
# results are:
# ADF Statistic, p-value, # of lags used, # of observations, critical values at 1,5,10%, maximized information creterion (AIC)
# https://chatgpt.com/share/681940fa-e644-800d-8bde-889d646d0170
adf_result_original = adfuller(iid_series)

# (b) Cumulative sum to simulate a non-stationary series
cumsum_series = np.cumsum(iid_series)
adf_result_cumsum = adfuller(cumsum_series)

# (c) Over-differentiated series: differentiate twice
diff_twice = np.diff(np.diff(cumsum_series))
adf_result_diff2 = adfuller(diff_twice)


# 5.2 Generate a time series that follows a sinusoidal function. This is a stationary
# series with memory.
# ----


# --- Utility Functions for Fractional Differencing ---
def get_weights_ffd(d, thresh=1e-5):
    """Get weights for fractional differentiation with fixed threshold"""
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thresh=1e-5):
    """Apply fixed-width fractional differencing"""
    w = get_weights_ffd(d, thresh)
    width = len(w) - 1
    output = []
    for i in range(width, len(series)):
        output.append(np.dot(w.T, series[i - width:i + 1])[0])
    return np.array(output)

# --- Generate Time Series ---
n = 500
t = np.arange(n)
sin_series = np.sin(2 * np.pi * t / 50)  # Sinusoidal (period 50)

# --- (a) ADF Test on Stationary Sinusoidal Series ---
adf_result_sin = adfuller(sin_series)

# --- (b) Make Non-Stationary Series (Shift + Cumsum) ---
shifted = sin_series + 5
cumsum_shifted = np.cumsum(shifted)
adf_result_cumsum_shifted = adfuller(cumsum_shifted)

# --- (ii) Expanding Window Fractional Differencing ---
min_d_expanding = None
d_values = np.arange(0.0, 1.1, 0.01)

for d in d_values:
    fd_series = frac_diff_ffd(cumsum_shifted, d, thresh=1e-2)
    if len(fd_series) > 0:
        p_value = adfuller(fd_series)[1]
        if p_value < 0.05 and min_d_expanding is None:
            min_d_expanding = d
            break  # Stop at first d with p < 0.05

# --- (iii) Fixed-Width FFD ---
min_d_ffd = None

for d in d_values:
    fd_series = frac_diff_ffd(cumsum_shifted, d, thresh=1e-5)
    if len(fd_series) > 0:
        p_value = adfuller(fd_series)[1]
        if p_value < 0.05 and min_d_ffd is None:
            min_d_ffd = d
            break  # Stop at first d with p < 0.05

# --- Print Results ---
# p-value: 0.0
#This confirms the series is stationary (we reject the null hypothesis of a unit root).
print("=== (a) Sinusoidal Series (stationary)===")
print(f"ADF Statistic: {adf_result_sin[0]:.4f}")
print(f"p-value: {adf_result_sin[1]:.4f}")

print("\n=== (b) Cumulative Shifted Series (non-stationary)===")
print(f"ADF Statistic: {adf_result_cumsum_shifted[0]:.4f}")
print(f"p-value: {adf_result_cumsum_shifted[1]:.4f}")

print("\n=== (ii) Expanding Window FracDiff (τ=1e-2) ===")
print(f"Minimum d with p < 0.05: {min_d_expanding}")

print("\n=== (iii) FFD (τ=1e-5) ===")
print(f"Minimum d with p < 0.05: {min_d_ffd}")

# --- Optional: Plotting the Time Series ---

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(sin_series)
plt.title("(a) Original Sinusoidal Series (Stationary with Memory)")

plt.subplot(3, 1, 2)
plt.plot(cumsum_shifted)
plt.title("(b) Cumulative Shifted Series (Non-Stationary with Memory)")

plt.subplot(3, 1, 3)
sample_fd = frac_diff_ffd(cumsum_shifted, min_d_ffd, thresh=1e-5)
plt.plot(sample_fd)
plt.title(f"(iii) FFD with d = {min_d_ffd:.2f} (Stationary)")

plt.tight_layout()
plt.show()


# 5.3  Take the seriesfrom exercise 2.b:
# (a) Fit the seriestoasinefunction. What istheR-squared?
# (b) Apply FFD(d=1). Fitthe seriestoasinefunction. What istheR-squared?
# (c) What value of d maximizes the R-squared of a sinusoidal fit on FFD(d).
# Why?
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# -- Sine Fitting Utility --
def sine_func(t, A, w, phi, C):
    return A * np.sin(w * t + phi) + C

def fit_sine_and_r2(series):
    t = np.arange(len(series))
    try:
        popt, _ = curve_fit(sine_func, t, series, p0=[1, 2 * np.pi / 50, 0, 0])
        fitted = sine_func(t, *popt)
        return r2_score(series, fitted)
    except Exception as e:
        print(e)
        return np.nan

# -- Generate cumulative sine series --
np.random.seed(42)
n = 500
t = np.arange(n)
sin_series = np.sin(2 * np.pi * t / 50)
cumsum_shifted = np.cumsum(sin_series + 5)

# -- (a) R² on non-stationary cumulative sine series --
r2_original = fit_sine_and_r2(cumsum_shifted)

# -- (b) Apply FFD with d=1 and fit --
fd_d1 = frac_diff_ffd(cumsum_shifted, d=1, thresh=1e-5)
r2_d1 = fit_sine_and_r2(fd_d1)

# -- (c) Sweep d values to find max R² --
d_values = np.arange(0.0, 2.01, 0.01)
r2_scores = []

for d in d_values:
    fd_series = frac_diff_ffd(cumsum_shifted, d, thresh=1e-5)
    if len(fd_series) > 0:
        r2 = fit_sine_and_r2(fd_series)
        r2_scores.append((d, r2))

best_d, best_r2 = max(r2_scores, key=lambda x: x[1])

print(f"(a) R² of sine fit on cumulative series: {r2_original:.4f}")
print(f"(b) R² after FFD(d=1): {r2_d1:.4f}")
print(f"(c) Best d for sine fit: {best_d:.2f}, with R² = {best_r2:.4f}")


# 5.4 Take the dollar bar series on E-mini S&P 500 futures. Using the code
# in Snippet 5.3, for some d ∈ [0,2], compute fracDiff_FFD(fracDiff
# _FFD(series,d),-d).What do you get? Why?
dollar_df = pd.read_csv('./dollar_df_2025_14_days.csv')
dollar_df['timestamp'] = pd.to_datetime(dollar_df['timestamp'])
dollar_df.set_index('timestamp', inplace=True)

def make_datetime_index_unique(df_with_dt_index):
    """
    Ensures a DataFrame's DatetimeIndex is unique by adding nanoseconds
    to duplicate timestamps. Modifies the DataFrame in place if a copy isn't made first.
    To be safe, operate on a copy if you need the original: df = df.copy()
    """
    if not isinstance(df_with_dt_index.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    if df_with_dt_index.index.is_unique:
        return df_with_dt_index

    # Calculate nanosecond offsets for duplicates using groupby and cumcount
    # cumcount provides 0 for the first item in a group, 1 for the second, etc.
    # We add this many nanoseconds. The first item in a duplicate group remains unchanged.
    # The 2nd gets +1ns, 3rd gets +2ns, etc.
    ns_offsets = pd.to_timedelta(
        df_with_dt_index.groupby(level=0).cumcount(), # Group by index values (timestamps)
        unit='us' # us = microsecond 1e-6 seconds. Even smaller is nano-seconds 1e-9
    )

    num_rows_with_duplicate_timestamps = df_with_dt_index.index.duplicated(keep=False).sum()
    print(f"Number of rows with duplicate timestamps: {num_rows_with_duplicate_timestamps} out of {len(df_with_dt_index)}")    
    # Add the offsets to the existing index
    df_with_dt_index.index = df_with_dt_index.index + ns_offsets
    
    return df_with_dt_index

dollar_df = make_datetime_index_unique(dollar_df)


def fracDiff_FFD(series,d,thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights_FFD(d,thres)
    width=len(w)-1
    
    #2) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):
                continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    
    df = pd.concat(df.values(), axis=1)
    
    return df


# This is not given in the book but still used in fracDiff_FFD
# It is dynamically deciding where to stop weights via thresh
def getWeights_FFD(d, thresh=1e-5):
    """
    Generate weights for fractional differencing using fixed threshold (FFD).
    Stops when weight magnitude falls below the threshold.
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


# fracDiff_FFD(dollar_df[['close']], 1)
# frac_diff_ffd(dollar_df[['close']], 0.95)

# ANSWER:recursive getWeights_FFD increasing the weights with -1
# eventually CRASHES the memory
# fracDiff_FFD(fracDiff_FFD(dollar_df[['close']], 1),-1)


# 5.5 Take the dollar bar series onE-mini S&P 500 futures.
# (a) Form a new seriesas acumulative sum of log-prices.


# (b) Apply FFD, with 𝜏 = 1E−5. Determine for what minimum d ∈ [0,2] the
# new series is stationary.

# Does not become stationary when d < 1
cumsum_log_series=  np.cumsum(np.log(dollar_df["close"]))

# Does become stationary when d = 0.69
log_series = np.log(dollar_df["close"])

d_values = np.arange(0.0, 2.01, 0.05)

for d in d_values:
    fd_series = frac_diff_ffd(cumsum_log_series, d, thresh=1e-5)
    print('d = ', d),
    if len(fd_series) > 0: 
        p_val = adfuller(fd_series)[1]
        print('p-value = ', p_val)
    if len(fd_series) > 0 and p_val < 0.05:
        print('Series is stationary with d = ', d)
        break

# (c) Compute the correlation of the frac diff series to the original(untransformed)
# series.

# len dollar_df["close"]  = 399
#len = 398
fd_series_cumsug_log = frac_diff_ffd(cumsum_log_series, 1, thresh=1e-5)
#len = 398
fd1_series_log = frac_diff_ffd(log_series, 1, thresh=1e-5)
#len = 321
fd95_series_log = frac_diff_ffd(log_series, 0.95, thresh=1e-5)

# ~ 1
correlation_fd_cumsug_log = np.corrcoef(dollar_df["close"][-len(fd_series_cumsug_log):], fd_series_cumsug_log)[0, 1]

# 0.115
correlation_fd1_log = np.corrcoef(dollar_df["close"][-len(fd1_series_log):], fd1_series_log)[0, 1]

# 0.181
correlation_fd95_log = np.corrcoef(dollar_df["close"][-len(fd95_series_log):], fd95_series_log)[0, 1]

# (d) Apply an Engel-Granger cointegration test on the original and frac diff series.
# Are they cointegrated? Why?
from statsmodels.tsa.stattools import coint

# Example: Using log prices and fractionally differenced log prices
log_series = np.log(dollar_df["close"]).dropna()
fd_log_series = frac_diff_ffd(log_series, d=0.95, thresh=1e-5).flatten()

# Align both series (must be same length)
min_len = min(len(log_series), len(fd_log_series))
log_series_aligned = log_series[-min_len:]
fd_log_series_aligned = fd_log_series[-min_len:]

# Apply Engle-Granger cointegration test
score, pvalue, _ = coint(log_series_aligned, fd_log_series_aligned)

print("Engle-Granger Cointegration Test")
print(f"Test statistic: {score}")
print(f"p-value: {pvalue}")

if pvalue < 0.05:
    print("✅ The series are cointegrated.")
else:
    print("❌ The series are not cointegrated.")

# Engle-Granger Cointegration Test
# Test statistic: -2.941199483582318
# p-value: 0.12485229074201137
# ❌ The series are not cointegrated.

# (e) Apply a Jarque-Bera normality teston the frac diff series.
import scipy.stats as stats

stats.jarque_bera(fd_log_series)

# 5.6 Take the frac diff series from exercise 5.
# (a) Apply a CUSUM filter (Chapter 2), where h is twice the standard deviation
# of the series.
def getTEvents(gRaw,h):
    tEvents,sPos,sNeg=[],0,0
    diff=gRaw.diff()
    for i in diff.index[1:]:
        sPos,sNeg=max(0,sPos+diff.loc[i]),min(0,sNeg+diff.loc[i])
        if sNeg<-h:
            sNeg=0;tEvents.append(i)
        elif sPos>h:
            sPos=0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

threshold = dollar_df["close"].diff().std()  # use standard deviation as threshold
t_events = getTEvents(dollar_df["close"], threshold * 2)

# (b) Sample a features matrix using the CUSUM timestamps
fd_log_series_99 = frac_diff_ffd(log_series, d=0.99, thresh=1e-5).flatten()
frac_diff = pd.Series(fd_log_series_99, index=dollar_df.index[-len(fd_log_series_99):])


# Align the frac_diff to cusum events
X = pd.DataFrame(index=t_events)
X['frac_diff'] = frac_diff.reindex(t_events)

# Add additional features (optional)
X['fast_ma'] = dollar_df['close'].rolling(5).mean().reindex(t_events)
X['slow_ma'] = dollar_df['close'].rolling(20).mean().reindex(t_events)
X['volatility'] = dollar_df['close'].pct_change().rolling(10).std().reindex(t_events)

X = X.dropna()

# (c) Form labels using the triple-barrier method
def applyPtSlOnT1(close,events,ptSl,molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_=events.loc[molecule]
    out=events_[['t1']].copy(deep=True)
    if ptSl[0]>0:
        pt=ptSl[0]*events_['trgt']
    else:
        pt=pd.Series(index=events.index) # NaNs
    
    if ptSl[1]>0:
        sl=-ptSl[1]*events_['trgt']
    else:
        sl=pd.Series(index=events.index) # NaNs
    
    for loc,t1 in events_['t1'].fillna(close.index[-1]).items():
        df0=close[loc:t1] # path prices
        df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss.
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking.
    
    return out

def getBins(events,close):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin']=np.sign(out['ret'])
    return out

# Estimate daily volatility
daily_vol = dollar_df['close'].pct_change().rolling('1D').std()
trgt = daily_vol.reindex(t_events, method='ffill')

# Create events
t1 = t_events + pd.Timedelta(days=5)
events = pd.DataFrame({
    't1': t1,
    'trgt': trgt,
    'side': 1  # assuming all long side, or you could bring in model-based side signal
})
# Apply triple barrier
barrier_timestamps = applyPtSlOnT1(dollar_df["close"], events, ptSl=[2, 2], molecule=events.index)
events['t1'] = barrier_timestamps.min(axis=1)

# Get labels
labels = getBins(events, dollar_df['close'])

# Align X and y
X = X.loc[labels.index.intersection(X.index)]
y = labels.loc[X.index, 'bin'].map({1: 1, 0: 0, -1: 0})  # meta-labeling format


# (d) Fit a Bagging Classifier with Sequential Bootstrapping and Uniqueness Weights
# (i) Sequential Bootstrapping of samples (from Chapter 4)



def seqBootstrap(indx, sampleLength, numCoEvents):
    wght = pd.Series(index=indx)
    phi = []
    while len(phi) < sampleLength:
        prob = (1. / numCoEvents.loc[indx])  # uniqueness-based probability
        prob /= prob.sum()
        pick = np.random.choice(indx, p=prob)
        phi.append(pick)
    return phi

def mpNumCoEvents(closeIdx,t1,molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    AVERAGE UNIQUENESS OF A LABEL 61
    Any event that starts before t1[molecule].max() impacts the count.
    '''
    #1) find events that span the period [molecule[0],molecule[-1]]
    t1=t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1=t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1=t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    
    #2) count events spanning a bar
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.items():count.loc[tIn:tOut]+=1.
    return count.loc[molecule[0]:t1[molecule].max()]

def mpSampleTW(t1,numCoEvents,molecule):
    # Derive average uniqueness over the event's lifespan
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].items():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght

# Compute numCoEvents
numCoEvents = mpNumCoEvents(closeIdx=dollar_df.index, t1=events['t1'], molecule=events.index)

# Compute average uniqueness weights
sample_weights = mpSampleTW(t1=events['t1'], numCoEvents=numCoEvents, molecule=events.index)

# Sequential bootstrap sampling
boot_indexes = seqBootstrap(X.index, sampleLength=int(len(X) * 0.8), numCoEvents=numCoEvents)

X_boot = X.loc[boot_indexes]
y_boot = y.loc[boot_indexes]
y_boot = y_boot.dropna()
w_boot = sample_weights.loc[boot_indexes].fillna(0)

# (ii) Train Bagging Classifier using sample weights
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Base estimator
tree = DecisionTreeClassifier(max_depth=5)

# Bagging classifier
bag = BaggingClassifier(estimator=tree, n_estimators=50, random_state=42)

# Fit using sample weights
bag.fit(X_boot, y_boot, sample_weight=w_boot)

# Evaluation
y_pred = bag.predict(X)

from sklearn.metrics import classification_report

print(classification_report(y.fillna(0), y_pred))
