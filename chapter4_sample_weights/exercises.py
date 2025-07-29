# 4.1 In Chapter 3, we denoted as t1 a pandas series of timestamps where the first
# barrier was touched, and the index was the timestamp of the observation. This
# was the output of thegetEvents function.
import lakeapi
import pandas as pd
import datetime
import numpy as np
import create_bars
# Example downlaoded from
# https://colab.research.google.com/drive/1E7MSUT8xqYTMVLiq_rMBLNcZmI_KusK3#scrollTo=hzlJ06LN35lt
# ------------------ Obtain Sample Data -------------------------------------- #
download_data = False

if download_data:
	lakeapi.use_sample_data(anonymous_access = True)

	btc_data = lakeapi.load_data(
			table="trades",
			start=datetime.datetime(2022, 10, 1),
			end=datetime.datetime(2022, 10, 2),
			symbols=["BTC-USDT"],
			exchanges=['BINANCE'],
	)
	btc_data.set_index('received_time', inplace = True)

	btc_data.to_csv('./btc_sample.csv')

# ------------------ Read Data -------------------------------------- #


# btc_data = pd.read_csv('../btc_sample.csv')
# btc_data['received_time'] = pd.to_datetime(btc_data['received_time'])
# btc_data.set_index('received_time', inplace = True)


print('Read Data ✅')

# 3.1 Form dollar bars for E-mini S&P 500 futures:
dollar_df_from_raw = False
if dollar_df_from_raw:
    dollar_df = create_bars.aggregate_data(btc_data, bar_type='dollar', threshold=8000000)


    dollar_df.to_csv('./data/dollar_df.csv')
else:
     dollar_df = pd.read_csv('./data/dollar_df_2025_14_days.csv')
     dollar_df['timestamp'] = pd.to_datetime(dollar_df['timestamp'])
     dollar_df.set_index('timestamp', inplace=True)

print('Dollars ready ✅')

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

# (a) Compute a t1 series on dollar bars derived from E-mini S&P 500 futures
# tickdata.
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


# Apply the CUSUM filter on the log of the close prices
threshold = dollar_df["close"].diff().std()  # use standard deviation as threshold
t_events = getTEvents(dollar_df["close"], threshold)
print(t_events)

# Add CUSUM event to 60 min, then find the closest data point in the original df to this
t1=dollar_df["close"].index.searchsorted(t_events+pd.Timedelta(minutes=60))
t1=t1[t1<dollar_df["close"].shape[0]]
t1=pd.Series(dollar_df["close"].index[t1],index=t_events[:t1.shape[0]]) # NaNs at end

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

## Calculate events parameter for 3barriers
# Estimate volatility
df_dollar_volatility = dollar_df['close'].pct_change().rolling('15min').std()
# Align volatility to event timestamps
trgt = df_dollar_volatility.reindex(t1.index, method='ffill')
events = pd.DataFrame({'t1': t1, 'trgt': trgt})
# We don't have the side defined yet. Hardcoding 1 for long
events["side"] = 1

# Apply original triple barrier function 
# barrier_timestamp df has index timestamp. Columns, t1 for vertical barrier, pt for profit take timestamp and sl for stop loss timestamp 
barrier_timestamps = applyPtSlOnT1(dollar_df["close"],events,ptSl=[1, 1],molecule=events.index)

barrier_touched = barrier_timestamps.min(axis=1)

print("len barrier_touched = ", len(barrier_touched))
print(barrier_touched.head())


# (b) ApplythefunctionmpNumCoEvents to compute the number of overlapping
# outcomes at each point intime.
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



co_events = mpNumCoEvents(
    closeIdx=dollar_df.index,
    t1=barrier_touched,
    molecule=events.index
)



# (c) Plot the time series of the number of concurrent labels on the primary axis,
# and the time series of exponentially weighted moving standard deviation of
# returnson the secondary axis.
import matplotlib.pyplot as plt

# Compute exponentially weighted volatility of returns
returns = dollar_df['close'].pct_change()
ewm_volatility = returns.ewm(span=50).std()

# Align to co_events index
ewm_volatility = ewm_volatility.reindex(co_events.index, method='ffill')

# Plot
fig, ax1 = plt.subplots(figsize=(14, 6))

# Primary axis: Number of concurrent labels
color = 'tab:blue'
ax1.set_xlabel('Time')
ax1.set_ylabel('Number of Concurrent Labels', color=color)
ax1.plot(co_events.index, co_events.values, color=color, label='Concurrent Labels')
ax1.tick_params(axis='y', labelcolor=color)

# Secondary axis: EWM volatility
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('EWM Volatility', color=color)
ax2.plot(ewm_volatility.index, ewm_volatility.values, color=color, linestyle='--', label='EWM Volatility')
ax2.tick_params(axis='y', labelcolor=color)

# Title and layout
plt.title('Concurrent Events vs. EWM Volatility')
fig.tight_layout()
plt.grid(True)
plt.show()


# (d) Produce a scatterplot of the number of concurrent labels (x-axis) and the
# exponentially weighted moving standard deviation of returns (y-axis). Can
# you appreciate a relationship?
import matplotlib.pyplot as plt

# Ensure both series are aligned
volatility_scatter = returns.ewm(span=50).std().reindex(co_events.index, method='ffill')

# Drop NaNs that might arise from alignment or initial volatility calculation
scatter_df = pd.DataFrame({
    'concurrent_labels': co_events,
    'ewm_volatility': volatility_scatter
}).dropna()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(scatter_df['concurrent_labels'], scatter_df['ewm_volatility'], alpha=0.6)
plt.xlabel('Number of Concurrent Labels')
plt.ylabel('EWM Volatility')
plt.title('Concurrent Labels vs. EWM Volatility')
plt.grid(True)
plt.show()

# 4.2 Using the function mpSampleTW, compute the average uniqueness of each label.
# What is the first-order serial correlation, AR(1), of this time series? Is it statisti-
# cally significant? Why?

def mpSampleTW(t1,numCoEvents,molecule):
    # Derive average uniqueness over the event's lifespan
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].items():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght

uniqueness = mpSampleTW(t1=barrier_touched, numCoEvents=co_events, molecule=events.index)

# Estimae AR(1)
import statsmodels.api as sm

# Drop NaNs if any
uniqueness = uniqueness.dropna()

# Compute lagged series
uniqueness_lag = uniqueness.shift(1).dropna()
uniqueness_trimmed = uniqueness.loc[uniqueness_lag.index]

# Run OLS regression (AR(1) model)
X = sm.add_constant(uniqueness_lag)
model = sm.OLS(uniqueness_trimmed, X).fit()

# P val 0.000 statistically significant
print(model.summary())


# 4.3 Fit arandom forest to a financial dataset where 
def get_moving_avg_signals(close, fast=5, slow=20):
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = np.sign(fast_ma - slow_ma)
    signal = signal.shift(1)  # avoid lookahead bias
    return signal

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

signals = get_moving_avg_signals(dollar_df["close"])

# Add side to events from the moving average signal
events_w_side = events.merge(signals, left_index=True, right_index=True)
barrier_timestamps_with_side = applyPtSlOnT1(dollar_df["close"], events_w_side, [20,1], events_w_side.index)

if "side"  in list(events_w_side.columns):
    del events_w_side["side"]
    events_w_side.rename(columns={"close": "side"}, inplace=True)

# Need to update t1 with barrier touch info obtained from applyPtSlOnT1
events_w_side['t1'] = barrier_timestamps_with_side.min(axis=1)

labels_with_side = getBins(events_w_side, dollar_df["close"])



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Build features
X = pd.DataFrame(index=dollar_df.index)
X['fast_ma'] = dollar_df['close'].rolling(5).mean()
X['slow_ma'] = dollar_df['close'].rolling(20).mean()
X['ma_diff'] = X['fast_ma'] - X['slow_ma']
X['volatility'] = dollar_df['close'].pct_change().rolling(10).std()
X = X.shift(1)  # Avoid look ahead bias

# 2. Build meta-labels (y)
# Match the index of your labels_with_side (triple-barrier output)
y = labels_with_side['bin'].copy()

# Map bins to {0,1} (meta-labels)
y = y.map({1:1, 0:0, -1:0})

# 3. Align X and y
Xy = X.loc[y.index]
X = Xy.dropna()
y = y.loc[X.index]

# X.to_csv("../X_chapter4_new.csv")
# y.to_csv("../y_chapter4_new.csv")


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 5. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Evaluate
# Accuracy
print(f"Train accuracy: {rf.score(X_train, y_train):.2f}")
# 0.7
print(f"Test accuracy: {rf.score(X_test, y_test):.2f}")

# (a) What isthe mean out-of-bag accuracy?
rf_oob = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True, bootstrap=True)
rf_oob.fit(X_train, y_train)
# 0.7155
print(f"Out-of-bag (OOB) accuracy: {rf_oob.oob_score_:.4f}")


# (b) What is the mean accuracy of k-fold cross-validation (without shuffling) on
# the same dataset?
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=False)
cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')
#0.637
print(f"Mean CV accuracy (no shuffling): {cv_scores.mean():.4f}")

# (c) Why is out-of-bag accuracy so much higher than cross-validation accuracy?
# Which one is more correct / less biased? What is the source of this bias?
print("\n(c) Why is OOB accuracy higher than CV accuracy?")
print("-" * 80)
print("OOB accuracy tends to be higher in time series or financial data due to temporal dependency.")
print("Cross-validation without shuffling avoids data leakage, making it more realistic and conservative.")
print("Therefore, CV accuracy is more correct and less biased in this context.")


# Consider you have applied meta-labels to events determined by a trend-following
# model. Suppose that two thirds of the labels are 0 and one third of the labels
# are 1.

## Find out actual label distribution
from collections import Counter

# Train without class weights
rf_unbalanced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_unbalanced.fit(X_train, y_train)

# Predictions
y_pred_unbalanced = rf_unbalanced.predict(X_test)
print("Prediction distribution (no class weight):", Counter(y_pred_unbalanced))



# Train with balanced class weights
rf_balanced = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_balanced.fit(X_train, y_train)

# Predictions with balanced weights
y_pred_balanced = rf_balanced.predict(X_test)

# Compare prediction distributions
print("Original label distribution:", Counter(y_test))
print("Predictions without class weights:", Counter(y_pred_unbalanced))
print("Predictions with balanced class weights:", Counter(y_pred_balanced))

