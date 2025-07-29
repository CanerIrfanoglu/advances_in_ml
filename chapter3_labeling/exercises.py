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


# btc_data = pd.read_csv('./btc_sample.csv')
# btc_data['received_time'] = pd.to_datetime(btc_data['received_time'])
# btc_data.set_index('received_time', inplace = True)


print('Read Data ✅')

# 3.1 Form dollar bars for E-mini S&P 500 futures:
dollar_df_from_raw = False
if dollar_df_from_raw:
    dollar_df = create_bars.aggregate_data(btc_data, bar_type='dollar', threshold=8000000)


    dollar_df.to_csv('./dollar_df.csv')
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

print('Dollar df index made unique ✅')

## (a) Apply a symmetric CUSUM filter (Chapter 2, Section 2.5.2.1) where the threshold isthe standard deviation of daily returns(Snippet 3.1).
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

# # Uncomment for plotting
# import matplotlib.pyplot as plt
# # Plot the results
# plt.figure(figsize=(10, 5))
# plt.plot(dollar_df.index, dollar_df["close"], label="Close Price")
# plt.scatter(t_events, dollar_df.loc[t_events]["close"], color="red", label="CUSUM Events", zorder=5)
# plt.title("CUSUM Filter Events on BTC Close Price")
# plt.xlabel("Timestamp")
# plt.ylabel("Price")
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


## (b) Use Snippet 3.4 on a pandas series t1,where minutes=1.

# Event to 60 min (this will act as vertical barrier), then find the closest data point in the original df to this
t1=dollar_df["close"].index.searchsorted(t_events+pd.Timedelta(minutes=60))
t1=t1[t1<dollar_df["close"].shape[0]]
t1=pd.Series(dollar_df["close"].index[t1],index=t_events[:t1.shape[0]]) # NaNs at end

# (c) On those sampled features, apply the triple-barrier method, where
# ptSl=[1,1]and t1 is the series you created in point 1.b.
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
        df0=(df0/close[loc]-1) * events_.at[loc,'side'] # path returns
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss.
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking.
    
    return out

## Calculate events parameter for 3barriers
# Estimate volatility
df_dollar_volatility = dollar_df['close'].pct_change().rolling('1D').std()
# Align volatility to event timestamps
trgt = df_dollar_volatility.reindex(t1.index, method='ffill')
events = pd.DataFrame({'t1': t1, 'trgt': trgt})
# We don't have the side defined yet. Hardcoding 1 for long
events["side"] = 1

# Apply original triple barrier function
barrier_timestamps = applyPtSlOnT1(dollar_df["close"],events,ptSl=[1, 1],molecule=events.index)


# (d) Apply getBinstogenerate thelabels.
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


labels = getBins(events, dollar_df["close"])


# 3.2 From exercise 1, useSnippet 3.8todrop rarelabels.

labels_binned = events.merge(labels, left_index=True, right_index=True)

def dropLabels(events,minPct=.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0=events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0]<3:
            print(df0.min(), df0.shape[0])
            break
        print('dropped label',df0.argmin(),df0.min())
        events=events[events['bin']!=df0.idxmin()]
    return events


labels_binned_dropped = dropLabels(labels_binned)
# 3.3 Adjust the getBins function (Snippet 3.5) to return a 0 whenever the vertical
# barrier isthe one touched first.

def getBinswithVertical(events, close):
    '''
    events: pd.DataFrame, must have columns: t1, trgt
        - index: event start time
        - t1: vertical barrier time
        - trgt: target volatility
    close: pd.Series, close prices
    '''
    # Align prices
    events_ = events.dropna(subset=['t1'])
    px = close.reindex(events_.index.union(events_['t1']).drop_duplicates()).sort_index()

    out = pd.DataFrame(index=events_.index)
    
    for event_time, event_row in events_.iterrows():
        # Define event parameters
        t1 = event_row['t1']
        trgt = event_row['trgt']
        event_price = px.loc[event_time]
        
        # Price path from event_time until t1 (inclusive)
        path = px.loc[event_time:t1]
        returns = (path / event_price) - 1  # normalized return path
        
        # Check if horizontal barrier is touched
        touched = None
        for time, ret in returns.items():
            if ret >= trgt:  # upper barrier
                touched = 1
                out.loc[event_time, 't1'] = time
                break
            elif ret <= -trgt:  # lower barrier
                touched = -1
                out.loc[event_time, 't1'] = time
                break
        
        # If touched is still None, it means vertical barrier hit first
        if touched is None:
            touched = 0
            out.loc[event_time, 't1'] = t1  # vertical barrier time
        
        out.loc[event_time, 'ret'] = returns.loc[out.loc[event_time, 't1']]
        out.loc[event_time, 'bin'] = touched

    return out


labels_binned_vertical = getBinswithVertical(events, dollar_df["close"])

# 3.4 Develop a trend-following strategy based on a popular technical analysis statistic
# (e.g.,crossing moving averages).For each observation, the model suggests a side,
# but not a size of the bet.
def get_moving_avg_signals(close, fast=5, slow=20):
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = np.sign(fast_ma - slow_ma)
    signal = signal.shift(1)  # avoid lookahead bias
    return signal

signals = get_moving_avg_signals(dollar_df["close"])

# 3.4 a) Derive meta-labels for ptSl=[1,2] and t1 where numDays=1. Use as
# trgt the dailys tandard deviation as computed by Snippet 3.1.
dollar_df_w_signals = dollar_df.merge(signals, left_index=True, right_index=True)

# Add side to events from the moving average signal
events_w_side = events.merge(signals, left_index=True, right_index=True)
del events_w_side["side"]
events_w_side.rename(columns={"close": "side"}, inplace=True)

barrier_timestamps_with_side = applyPtSlOnT1(dollar_df["close"], events_w_side, [1,2], events_w_side.index)
labels_with_side = getBins(events_w_side, dollar_df["close"])


# 3.4 b) Train a random forest to decide whether to trade or not. Note: The decision
# is whether to trade or not, {0,1}, since the underlying model (the crossing
# moving average) has decided the side, {−1,1}.
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
X = X.shift(1)  # Avoid lookahead bias

# 2. Build meta-labels (y)
# Match the index of your labels_with_side (triple-barrier output)
y = labels_with_side['bin'].copy()

# Map bins to {0,1} (meta-labels)
y = y.map({1:1, 0:0, -1:0})

# 3. Align X and y
Xy = X.loc[y.index]
X = Xy.dropna()
y = y.loc[X.index]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 5. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Evaluate
# Accuracy
print(f"Train accuracy: {rf.score(X_train, y_train):.2f}")
print(f"Test accuracy: {rf.score(X_test, y_test):.2f}")

# Predictions
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
# Precision, Recall, F1
print("\nTrain metrics:")
print(f"Precision: {precision_score(y_train, y_pred_train):.2f}")
print(f"Recall: {recall_score(y_train, y_pred_train):.2f}")
print(f"F1 Score: {f1_score(y_train, y_pred_train):.2f}")

print("\nTest metrics:")
print(f"Precision: {precision_score(y_test, y_pred_test):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_test):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_test):.2f}")

# Optional: Confusion Matrix
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))



# 3.5 Develop a mean-reverting strategy based on Bollinger bands. For each observa-
# tion, the model suggests aside, but not a size of the bet.

def get_bollinger_signals(close, window=20, num_std=2):
    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    
    signal = np.where(close > upper_band, -1, np.nan)  # Short if above upper
    signal = np.where(close < lower_band, 1, signal)   # Long if below lower
    signal = pd.Series(signal, index=close.index)
    signal = signal.ffill()  # Maintain position until changed
    signal = signal.shift(1)  # Avoid lookahead bias
    return signal

bollinger_signals = get_bollinger_signals(dollar_df["close"])
bollinger_signals.name = 'bollinger_signal'

# import matplotlib.pyplot as plt

# # Plotting
# plt.figure(figsize=(14, 8))

# # Plot Close price
# plt.plot(dollar_df.index, dollar_df['close'], label='Close Price', color='blue', linewidth=1.5)

# # Plot Bollinger Bands
# plt.plot(dollar_df.index, dollar_df['close'].rolling(20).mean(), label='Rolling Mean', color='orange', linestyle='--')
# plt.plot(dollar_df.index, dollar_df['close'].rolling(20).mean() + 2 * dollar_df['close'].rolling(20).std(), 
#          label='Upper Band', color='red', linestyle='-.')
# plt.plot(dollar_df.index, dollar_df['close'].rolling(20).mean() - 2 * dollar_df['close'].rolling(20).std(), 
#          label='Lower Band', color='green', linestyle='-.')

# # Plot Buy/Sell signals
# buy_signals = bollinger_signals == 1
# sell_signals = bollinger_signals == -1

# # Plot buy signals (below lower band)
# plt.scatter(dollar_df.index[buy_signals], dollar_df['close'][buy_signals], marker='^', color='green', label='Buy Signal', alpha=1)

# # Plot sell signals (above upper band)
# plt.scatter(dollar_df.index[sell_signals], dollar_df['close'][sell_signals], marker='v', color='red', label='Sell Signal', alpha=1)

# # Labels and legend
# plt.title('Bollinger Bands with Buy/Sell Signals')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend(loc='best')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()

# plt.show()

# (a) Derive meta-labels for ptSl=[0,2] and t1 where numDays=1. Use as
# trgt the daily standard deviation as computed by Snippet 3.1.
dollar_df_w_bollinger_signals = dollar_df.merge(bollinger_signals, left_index=True, right_index=True)

# Add side to events from the moving average signal
events_w_side_bollinger = events.merge(bollinger_signals, left_index=True, right_index=True)
del events_w_side_bollinger["side"]
events_w_side_bollinger.rename(columns={"bollinger_signal": "side"}, inplace=True)

barrier_timestamps_with_side_bollinger = applyPtSlOnT1(dollar_df["close"], events_w_side_bollinger, [0,2], events_w_side_bollinger.index)
labels_with_side_bollinger = getBins(events_w_side_bollinger, dollar_df["close"])


# (b) Train a random forest to decide whether to trade or not. Use as fea-
# tures: volatility, serial correlation, and the crossing moving averages from
# exercise 2.
# 1. Build features
X = pd.DataFrame(index=dollar_df.index)
X['fast_ma'] = dollar_df['close'].rolling(5).mean()
X['slow_ma'] = dollar_df['close'].rolling(20).mean()
X['ma_diff'] = X['fast_ma'] - X['slow_ma']
X['volatility'] = dollar_df['close'].pct_change().rolling(10).std()
X['serial_corr'] = dollar_df["close"].pct_change().rolling(5).apply(lambda x: x.autocorr(), raw=False)

X = X.shift(1)  # Avoid lookahead bias

# 2. Build meta-labels (y)
# Match the index of your labels_with_side (triple-barrier output)
y = labels_with_side_bollinger['bin'].copy()

# Map bins to {0,1} (meta-labels)
y = y.map({1:1, 0:0, -1:0})

# 3. Align X and y
Xy = X.loc[y.index]
X = Xy.dropna()
y = y.loc[X.index]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 5. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# 3.5 c What is the accuracy of predictions from the primary model (i.e., if the sec-
# ondary model does not filter the bets)? What are the precision, recall, and
#F1-scores?
primary_preds = np.where(labels_with_side_bollinger['bin'].loc[X_test.index] == 1, 1, 0)  # Trade only if profit, else 0


print("Primary model performance:")
print(f"Accuracy: {accuracy_score(y_test, primary_preds):.2f}")
print(f"Precision: {precision_score(y_test, primary_preds):.2f}")
print(f"Recall: {recall_score(y_test, primary_preds):.2f}")
print(f"F1 Score: {f1_score(y_test, primary_preds):.2f}")



# 3.5 d 
# 6. Evaluate
# Accuracy
print(f"Train accuracy: {rf.score(X_train, y_train):.2f}")
print(f"Test accuracy: {rf.score(X_test, y_test):.2f}")

# Predictions
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
# Precision, Recall, F1
print("\nTrain metrics:")
print(f"Precision: {precision_score(y_train, y_pred_train):.2f}")
print(f"Recall: {recall_score(y_train, y_pred_train):.2f}")
print(f"F1 Score: {f1_score(y_train, y_pred_train):.2f}")

print("\nTest metrics:")
print(f"Precision: {precision_score(y_test, y_pred_test):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_test):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_test):.2f}")

# Optional: Confusion Matrix
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))


# (c) What is the accuracy of predictions from the primary model (i.e., if the sec-
# ondary model does not filter the bets)? What are the precision, recall, and
# F1-scores?
# (d) Whatistheaccuracyofpredictionsfromthesecondarymodel?Whatarethe
# precision, recall, and F1-scores?




