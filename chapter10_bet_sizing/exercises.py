import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Below function getSignal and it's helpers are about processing a series of signals over time, averaging signals from concurrently open bets, and discretizing them.
def getSignal(events,stepSize,prob,pred,numClasses,numThreads,**kargs):
    # get signals from predictions
    if prob.shape[0]==0:
        return pd.Series()

    #1) generate signals from multinomial classification (one-vs-rest, OvR)
    signal0=(prob-1./numClasses)/(prob*(1.-prob))**.5 # t-value of OvR
    signal0=pred*(2*stats.norm.cdf(signal0)-1) # signal=side*size
    if 'side' in events:
        signal0*=events.loc[signal0.index,'side'] # meta-labeling

    #2) compute average signal among those concurrently open
    df0=signal0.to_frame('signal').join(events[['t1']],how='left')
    df0=avgActiveSignals(df0,numThreads)
    signal1=discreteSignal(signal0=df0,stepSize=stepSize)
    return signal1

def discreteSignal(signal0,stepSize):
    # discretize signal
    signal1=(signal0/stepSize).round()*stepSize # discretize
    signal1[signal1>1]=1 # cap
    signal1[signal1<-1]=-1 # floor
    return signal1

def avgActiveSignals(signals,numThreads):
    # compute the average signal among those active
    #1) time points where signals change (either one starts or one ends)
    tPnts=set(signals['t1'].dropna().values)
    tPnts=tPnts.union(signals.index.values)
    tPnts=list(tPnts);tPnts.sort()
    out=mpPandasObj(mpAvgActiveSignals,('molecule',tPnts),numThreads,signals=signals)
    return out

def mpAvgActiveSignals(signals,molecule):
    '''
    At time loc, average signal among those still active.
    Signal is active if:
    a) issued before or at loc AND
    b) loc before signal's endtime, or endtime is still unknown (NaT).
    '''
    out=pd.Series()
    for loc in molecule:
        df0=(signals.index.values<=loc)&((loc<signals['t1'])|pd.isnull(signals['t1']))
        act=signals[df0].index
    if len(act)>0:
        out[loc]=signals.loc[act,'signal'].mean()
    else:
        out[loc]=0 # no signals active at this time
    return out



# 10.1 Using the formulation in Section10.3, plot the bet size (m) as a
# function of the maximum predicted probability (̃p) when ‖X‖= 2,3,…,10.
def calculate_bet_size_m(p_tilde, num_classes):
    """
    Calculates the bet size 'm' based on the formula derived from getSignal's logic
    and Section 10.3's bet sizing.
    p_tilde: maximum predicted probability (for the predicted class)
    num_classes: number of possible outcomes ||X||
    """
    if p_tilde <= 0 or p_tilde >= 1: # Probabilities must be between 0 and 1 (exclusive for denominator)
        return np.nan # Or handle as per specific requirements for edge cases
    if p_tilde == 1.0/num_classes: # Avoid division by zero if p_tilde is exactly the null hypothesis
        z = 0.0
    else:
        # This is the z-score or t-value used for sizing in getSignal
        # z = (p_tilde - (1.0 / num_classes)) / np.sqrt(p_tilde * (1.0 - p_tilde))
        # The formula for z in figure 10.1 is sqrt(N) * (p - p0) / sqrt(p0 * (1-p0)) for binomial
        # Or for proportion p against p0: (p-p0) / sqrt(p0(1-p0)/N)
        # For practical bet sizing based on probability p_tilde vs 1/num_classes (random guess):
        # A common way to get a z-like score for bet sizing:
        numerator = p_tilde - (1.0 / num_classes)
        denominator = np.sqrt(p_tilde * (1.0 - p_tilde)) # this is a component of std error of p_tilde
        
        # If p_tilde is very close to 0 or 1, denominator can be close to 0.
        if np.isclose(denominator, 0):
            # If p_tilde is very high (close to 1) and num_classes is small (e.g. 2, so 1/num_classes=0.5)
            # then numerator is positive, we want high confidence.
            # If p_tilde is very low (close to 0), then this p_tilde was NOT the max prob for a class.
            # This function assumes p_tilde IS the max probability for the *chosen* class.
            # So, p_tilde should generally be > 1/num_classes.
             if numerator > 0:
                z = np.inf
             elif numerator < 0: # p_tilde < 1/num_classes, should not happen if p_tilde is max and we pick this class
                z = -np.inf
             else:
                z = 0
        else:
            z = numerator / denominator

    # Bet size m = side * (2*CDF(z) - 1). Assume side = +1 since p_tilde is prob of *predicted* class.
    m = 2 * stats.norm.cdf(z) - 1
    return m



# Generate a range of p_tilde values (maximum predicted probabilities)
# p_tilde must be > 1/num_classes for a positive bet on that class.
# And p_tilde <= 1.
# Let's create p_tilde values from just above random guess up to nearly 1.
p_tilde_values = np.linspace(0.01, 0.99, 200) # General range


plt.figure(figsize=(12, 8))

for num_classes in range(2, 11): # ||X|| = 2, 3, ..., 10
    bet_sizes = []
    # Filter p_tilde values relevant for this num_classes
    # p_tilde should be at least 1/num_classes to be the "max" probability for a chosen class
    # and to make sense in the formula (p_tilde - 1/num_classes) to be positive.
    # If p_tilde < 1/num_classes, it means this class (whose prob is p_tilde) should not have been chosen.
    # The formula for 'm' is for the size of the bet on the class whose probability IS p_tilde.
    # So, we are interested when p_tilde > 1/num_classes.
    
    current_p_tilde_values = [p for p in p_tilde_values if p > (1.0/num_classes) + 1e-6] # Add epsilon to avoid p_tilde = 1/N issues
    if not current_p_tilde_values and p_tilde_values[0] <= (1.0/num_classes): # If all p_tilde are too low
        # For high num_classes, 1/num_classes is small. np.linspace(0.01, 0.99, 200) should cover it.
        # If 1/num_classes is e.g. 0.5 (for num_classes=2), we need p_tilde > 0.5.
        # If 1/num_classes is e.g. 0.1 (for num_classes=10), we need p_tilde > 0.1.
        # We should generate p_tilde relative to 1/num_classes
        specific_p_range = np.linspace((1.0/num_classes) + 1e-6, 0.999, 100)

    else:
        specific_p_range = current_p_tilde_values


    for p_t in specific_p_range:
        # Ensure p_t is a valid probability for being the max
        if p_t < (1.0/num_classes): # This p_t couldn't be the max prob for a class we bet on
            bet_sizes.append(0) # Or np.nan, or skip
            continue
        m = calculate_bet_size_m(p_t, num_classes)
        bet_sizes.append(m)
    
    if specific_p_range and bet_sizes: # Check if lists are not empty
        plt.plot(specific_p_range, bet_sizes, label=f'||X|| = {num_classes}')

plt.xlabel('Maximum Predicted Probability (p̃)')
plt.ylabel('Bet Size (m)')
plt.title('Bet Size (m) vs. Max Predicted Probability (p̃) for different ||X||')
plt.legend()
plt.grid(True)
plt.ylim(-0.05, 1.05) # Bet size m is typically scaled between 0 and 1 for the chosen side
plt.axhline(0, color='black', linewidth=0.5)
plt.show()



# --- Sequential substitute for avgActiveSignals ---
def mpAvgActiveSignals_sequential_batch(signals_df, t_points_molecule):
    """
    Helper function for sequential avgActiveSignals.
    Processes a batch of t_points.
    'signals_df' should have a DatetimeIndex and columns 'signal' and 't1'.
    't_points_molecule' is a list of timestamps to calculate average active signals for.
    """
    out_signals = pd.Series(index=t_points_molecule, dtype=float)
    for loc in t_points_molecule:
        # Find signals active at time 'loc'
        # Active if: signal_start_time <= loc AND (loc < signal_end_time OR signal_end_time is NaT)
        active_mask = (signals_df.index <= loc) & \
                      ((signals_df['t1'].isna()) | (loc < signals_df['t1']))
        
        active_signals = signals_df.loc[active_mask, 'signal']
        
        if not active_signals.empty:
            out_signals[loc] = active_signals.mean()
        else:
            out_signals[loc] = 0.0
    return out_signals

def avgActiveSignals_sequential(signals_df, numThreads=1): # numThreads is ignored here
    """
    Sequential version of avgActiveSignals.
    'signals_df' should have a DatetimeIndex and columns 'signal' and 't1'.
    """
    # 1) Determine all unique time points where signals start or end, or exist.
    t_points = set(signals_df.index.tolist())
    if 't1' in signals_df.columns:
        t_points.update(signals_df['t1'].dropna().tolist())
    
    sorted_t_points = sorted(list(t_points))
    
    if not sorted_t_points:
        print("No signals found.")
        return pd.Series(dtype=float)

    # In a truly large dataset, you might process sorted_t_points in chunks.
    # For this exercise, direct processing is fine.
    # Using tqdm for progress indication as this can be slow for many points.
    print(f"Calculating average active signals for {len(sorted_t_points)} points sequentially...")
    
    # The original mpAvgActiveSignals was designed to be called by mpPandasObj
    # which passes 'molecule' (a subset of t_points).
    # Here, we simulate that by passing all sorted_t_points as the 'molecule'.
    
    # Prepare for mpAvgActiveSignals_sequential_batch
    # Ensure signals_df index is DatetimeIndex for proper comparison with loc
    # signals_df.index = pd.to_datetime(signals_df.index)
    # if 't1' in signals_df.columns:
    #     signals_df['t1'] = pd.to_datetime(signals_df['t1'])

    # Using the batch helper directly
    out_series = pd.Series(dtype=float)
    # Process in batches if memory/speed is an issue, otherwise one batch.
    # For simplicity, let's process as one batch here, but use tqdm on the loop inside the helper.
    # So, effectively, we can just call the inner logic.

    averaged_signals = pd.Series(index=pd.to_datetime(sorted_t_points), dtype=float)
    
    for loc in tqdm(averaged_signals.index, desc="Averaging Signals"):
        active_mask = (signals_df.index <= loc) & \
                      ((signals_df['t1'].isna()) | (loc < signals_df['t1']))
        active_s = signals_df.loc[active_mask, 'signal']
        if not active_s.empty:
            averaged_signals[loc] = active_s.mean()
        else:
            averaged_signals[loc] = 0.0
            
    return averaged_signals

# 10.2 Draw 10,000 random numbers from a uniform distribution with bounds U[.5,1.].
print("--- Exercise 10.2 ---")

np.random.seed(42) # for reproducibility
p_tilde_samples_a = np.random.uniform(low=0.5, high=1.0, size=10000)

# (a) Compute the bet sizes m for ||X||= 2.
num_classes_a = 2
bet_sizes_m_a = [calculate_bet_size_m(p_t, num_classes_a) for p_t in p_tilde_samples_a]
bet_sizes_m_a = pd.Series(bet_sizes_m_a) # Convert to Series for easier handling later


# (b) Assign 10,000 consecutive calendar days tothebet sizes.
# Let's start from a fixed date for reproducibility.
start_date_b = pd.to_datetime('2020-01-01')
# Create a DatetimeIndex of 10,000 CALENDAR days (freq='D')
dates_b = pd.date_range(start=start_date_b, periods=10000, freq='D')

# Assign these dates as index to the bet sizes
# This Series now represents our 'signal' at each date.
signals_at_date_b = pd.Series(bet_sizes_m_a.values, index=dates_b)
print(f"\n(b) Assigned bet sizes to {len(dates_b)} consecutive calendar days, starting {dates_b[0]}.")
print("Signals at date head:\n", signals_at_date_b.head())


# (c) Draw 10,000 random numbers from a uniform distribution with bounds U[1,25].
# These will be the holding periods (durations) for each signal/bet.
np.random.seed(123) # for reproducibility
holding_periods_c = np.random.uniform(low=1, high=25, size=10000)
# Since holding periods are usually integers (days), let's round them.
holding_periods_c = np.round(holding_periods_c).astype(int)
print(f"\n(c) Drawn {len(holding_periods_c)} random holding periods (1 to 25 days).")
print("First 5 holding periods:", holding_periods_c[:5])

# (d) Form a pandas series indexed by the dates in 2.b, and with values equal
# to the index shifted forward the number of days in 2.c. This is a t1 object.
# 't1' represents the time when the bet/signal expires or the barrier is touched.

t1_values_d = []
for i in range(len(dates_b)):
    start_of_signal = dates_b[i]
    duration = pd.Timedelta(days=holding_periods_c[i])
    end_of_signal = start_of_signal + duration
    t1_values_d.append(end_of_signal)

t1_series_d = pd.Series(t1_values_d, index=dates_b)
print(f"\n(d) Formed t1 series (signal end times).")
print("t1 series head:\n", t1_series_d.head())

# Create the 'events' DataFrame structure needed for avgActiveSignals
# It needs an index (signal start time), a 'signal' column (bet size), and a 't1' column (signal end time).
events_df_e = pd.DataFrame({
    'signal': signals_at_date_b, # These are the bet sizes 'm' from part (a)
    't1': t1_series_d
})
events_df_e.index.name = 'signal_start_time' # Optional: name the index

print("\nEvents DataFrame for averaging (head):")
print(events_df_e.head())

# (e) Compute the resulting average active bets, following Section 10.4.
# We use our sequential substitute: avgActiveSignals_sequential
average_active_bets_e = avgActiveSignals_sequential(events_df_e)

print(f"\n(e) Computed average active bets over time.")
print("Average active bets series (head):\n", average_active_bets_e.head())
print("Average active bets series (tail):\n", average_active_bets_e.tail())
print("Descriptive stats for average_active_bets_e:\n", average_active_bets_e.describe())

# Optional: Plot the average active bets
plt.figure(figsize=(12, 6))
average_active_bets_e.plot(title='Average Active Bets Over Time')
plt.xlabel('Date')
plt.ylabel('Average Bet Size')
plt.grid(True)
plt.show()

# --- Exercise 10.3 ---
print("\n--- Exercise 10.3 ---")
# Using the t1 object from exercise 2.d (which is part of events_df_for_exercises)

# To implement 10.3, we need c_t,l and c_t,s.
# As discussed, all signals in events_df_for_exercises['signal'] are positive (0 to 1).
# We'll treat them as "long" signals. So, c_t,s will be 0.
# c_t,l will be the count of active signals.
def get_active_signal_stats(signals_df, bet_type_col=None):
    """
    Calculates the average signal and the count of active signals over time.
    'signals_df' should have a DatetimeIndex and columns 'signal' and 't1'.
    If 'bet_type_col' is provided (e.g. a column indicating 'long' or 'short'),
    it will also return counts for each type.
    Returns:
        - avg_signals (Series)
        - count_total_signals (Series)
        - count_long_signals (Series, if bet_type_col specified and has 'long')
        - count_short_signals (Series, if bet_type_col specified and has 'short')
    """
    t_points = set(signals_df.index.tolist())
    if 't1' in signals_df.columns:
        t_points.update(signals_df['t1'].dropna().tolist())
    sorted_t_points = sorted(list(t_points))
    if not sorted_t_points:
        return pd.Series(dtype=float), pd.Series(dtype=int), pd.Series(dtype=int), pd.Series(dtype=int)

    datetime_index = pd.to_datetime(sorted_t_points)
    avg_signals = pd.Series(index=datetime_index, dtype=float)
    count_total_signals = pd.Series(index=datetime_index, dtype=int)
    count_long_signals = pd.Series(index=datetime_index, dtype=int) if bet_type_col else None
    count_short_signals = pd.Series(index=datetime_index, dtype=int) if bet_type_col else None
    
    print(f"Calculating active signal stats for {len(sorted_t_points)} points sequentially...")
    for loc in tqdm(datetime_index, desc="Processing Time Points"):
        active_mask = (signals_df.index <= loc) & \
                      ((signals_df['t1'].isna()) | (loc < signals_df['t1']))
        active_df_slice = signals_df.loc[active_mask]
        
        if not active_df_slice.empty:
            avg_signals[loc] = active_df_slice['signal'].mean()
            count_total_signals[loc] = active_df_slice['signal'].count()
            if bet_type_col:
                # Assuming 'long' signals have positive 'signal' values, 'short' have negative
                # Or, if a specific 'bet_type' column exists, use that.
                # For this exercise, we'll infer from the sign of 'signal'.
                if count_long_signals is not None:
                    count_long_signals[loc] = (active_df_slice['signal'] > 0).sum()
                if count_short_signals is not None:
                    count_short_signals[loc] = (active_df_slice['signal'] < 0).sum()
        else:
            avg_signals[loc] = 0.0
            count_total_signals[loc] = 0
            if bet_type_col:
                if count_long_signals is not None: count_long_signals[loc] = 0
                if count_short_signals is not None: count_short_signals[loc] = 0
                
    if bet_type_col:
        return avg_signals, count_total_signals, count_long_signals, count_short_signals
    else:
        return avg_signals, count_total_signals
    
_, _, concurrent_long_bets_series, concurrent_short_bets_series = \
    get_active_signal_stats(events_df_e, bet_type_col='signal') # Pass a dummy col to trigger type counts


max_concurrent_long_c_l_bar = concurrent_long_bets_series.max()
max_concurrent_short_c_s_bar = concurrent_short_bets_series.max()

print(f"(a) Maximum number of concurrent long bets (c̄_l): {max_concurrent_long_c_l_bar}")
print(f"(b) Maximum number of concurrent short bets (c̄_s): {max_concurrent_short_c_s_bar}")


# (c) Derive the bet size as m_t = c_t,l * (1/c̄_l) - c_t,s * (1/c̄_s)
# c_t,l is the value from c_l_series at time t
# c_t,s is the value from c_s_series at time t

# Ensure c_l_bar and c_s_bar are not zero to avoid division by zero.
# If max is 0, it means that type of bet never occurred, or no signals were active.
# In such a case, the contribution from that term should be 0.
term_l = concurrent_long_bets_series * (1 / max_concurrent_long_c_l_bar if max_concurrent_long_c_l_bar > 0 else 0)
term_s = concurrent_short_bets_series * (1 / max_concurrent_short_c_s_bar if max_concurrent_short_c_s_bar > 0 else 0)
m_t_budget = term_l - term_s

print("\n(c) Bet size series m_t (budgeting approach) - Head:")
print(m_t_budget.head())
print("\nDescriptive stats for m_t_budget:")
print(m_t_budget.describe())

plt.figure(figsize=(12, 6))
m_t_budget.plot(title='Bet Size Over Time (Budgeting Approach - Ex 10.3)')
plt.xlabel('Date')
plt.ylabel('Bet Size (m_t)')
plt.grid(True)
plt.show()

# --- Exercise 10.4 ---
print("\n--- Exercise 10.4 ---")
# Using the t1 object from exercise 2.d (from events_df_for_exercises or events_df_mixed)

# (a) Compute the series c_t = c_t,l - c_t,s
# We already have c_l_series and c_s_series from the mixed events for 10.3
c_t_series = concurrent_long_bets_series - concurrent_short_bets_series
print("(a) Net concurrent bets series (c_t) - Head:")
print(c_t_series.head())

# (b) Fit a mixture of two Gaussians on {c_t}.
# We need to reshape c_t_series for GaussianMixture input (it expects 2D array)
from sklearn.mixture import GaussianMixture
c_t_values_for_gmm = c_t_series.dropna().values.reshape(-1, 1)

if len(c_t_values_for_gmm) > 1 : # Need at least 2 samples for GMM
    gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='spherical') # Spherical for simpler variance
    gmm.fit(c_t_values_for_gmm)
    print(f"\n(b) Fitted Gaussian Mixture Model with 2 components.")
    print(f"  Means: {gmm.means_.flatten()}")
    print(f"  Covariances (variances for spherical): {gmm.covariances_.flatten()}") # Variances for spherical
    print(f"  Weights: {gmm.weights_.flatten()}")

    # (c) Derive the bet size m_t = { ... F[c_t] ... } where F[x] is the CDF of the fitted GMM.
    # The CDF for a GMM is weights[0]*CDF_component1 + weights[1]*CDF_component2
    def gmm_cdf(x_values, gmm_model):
        # Ensure x_values is a 2D array for predict_proba and for stats.norm.cdf
        x_values_2d = np.array(x_values).reshape(-1, 1)
        
        # cdf_probs = np.zeros(len(x_values_2d))
        # for i in range(gmm_model.n_components):
        #     # For spherical, gmm_model.covariances_ are variances. Std dev is sqrt(variance).
        #     std_dev = np.sqrt(gmm_model.covariances_[i])
        #     cdf_probs += gmm_model.weights_[i] * stats.norm.cdf(x_values_2d.flatten(),
        #                                                         loc=gmm_model.means_[i, 0],
        #                                                         scale=std_dev)
        # return cdf_probs

        # Simpler: Use predict_proba to get P(component), but that's not CDF.
        # We need the actual CDF.
        # The CDF of a mixture is sum of (weight_i * CDF_i)
        cdf_values = np.zeros_like(x_values, dtype=float)
        for i in range(len(x_values)):
            val = x_values[i]
            cumm_prob = 0
            for j in range(gmm_model.n_components):
                mean = gmm_model.means_[j, 0]
                # For spherical covariance_type, gmm.covariances_ stores the variance.
                std_dev = np.sqrt(gmm_model.covariances_[j])
                cumm_prob += gmm_model.weights_[j] * stats.norm.cdf(val, loc=mean, scale=std_dev)
            cdf_values[i] = cumm_prob
        return cdf_values

    F_ct = pd.Series(gmm_cdf(c_t_series.values, gmm), index=c_t_series.index)
    F_0 = gmm_cdf(np.array([0]), gmm)[0] # CDF at x=0

    m_t_cdf = pd.Series(index=c_t_series.index, dtype=float)
    for t, ct_val in c_t_series.items():
        if ct_val >= 0:
            if (1 - F_0) == 0 : # Avoid division by zero
                 m_t_cdf[t] = 1 if (F_ct[t] - F_0) > 0 else 0 # or some other handling
            else:
                m_t_cdf[t] = (F_ct[t] - F_0) / (1 - F_0)
        else: # ct_val < 0
            if F_0 == 0: # Avoid division by zero
                m_t_cdf[t] = -1 if (F_ct[t] - F_0) < 0 else 0 # or some other handling
            else:
                m_t_cdf[t] = (F_ct[t] - F_0) / F_0
    
    # Clip values to be within [-1, 1] as ratios can sometimes exceed due to numerical precision
    m_t_cdf = m_t_cdf.clip(-1, 1)

    print("\n(c) Bet size series m_t (CDF approach) - Head:")
    print(m_t_cdf.head())
    print("\nDescriptive stats for m_t_cdf:")
    print(m_t_cdf.describe())

    plt.figure(figsize=(12, 6))
    m_t_cdf.plot(title='Bet Size Over Time (CDF of GMM Approach - Ex 10.4)')
    plt.xlabel('Date')
    plt.ylabel('Bet Size (m_t)')
    plt.grid(True)
    plt.show()

    # (d) Explain how this series {m_t} (CDF approach) differs from the bet size series computed in exercise 3 (budgeting approach).
    print("\n(d) Comparison of Bet Sizing Methods:")
    print("   - Budgeting Approach (Ex 10.3):")
    print("     - Bet size is proportional to the current number of active long/short signals relative to the *historical maximum* number of concurrent long/short signals.")
    print("     - It's a linear scaling based on 'capacity utilization'.")
    print("     - If c_t,l is half of the max_concurrent_long, the long component of the bet is 0.5.")
    print("     - It doesn't directly consider the overall distribution of net signals (c_t), only the current count vs max count.")
    print("   - CDF of GMM Approach (Ex 10.4):")
    print("     - Bet size is based on the *percentile rank* of the current net signal strength (c_t) within its own historical distribution (modeled by GMM).")
    print("     - F[c_t] tells you what percentage of historical net signals were weaker than or equal to the current c_t.")
    print("     - The formula (F[c_t] - F[0]) / (1 - F[0]) (for c_t >= 0) normalizes this percentile rank relative to the probability of any positive signal.")
    print("     - This results in a non-linear, S-shaped response to c_t. Small changes in c_t around the median of its distribution can lead to larger changes in bet size than changes at the extremes.")
    print("     - It's sensitive to how 'unusual' or 'extreme' the current net signal strength is compared to its typical behavior.")
    print("   - Key Differences:")
    print("     - Linearity vs. Non-linearity: Budgeting is linear in c_t,l and c_t,s (up to their max). CDF is non-linear in c_t.")
    print("     - Information Used: Budgeting uses max historical counts. CDF uses the full distribution of historical net signals.")
    print("     - Sensitivity: CDF method might be more sensitive to small changes in c_t if c_t is in a steep part of its CDF, while budgeting is uniformly sensitive until max capacity.")
else:
    print("\n(b) Not enough data points in c_t_series to fit Gaussian Mixture Model for Exercise 10.4.")
    print("(c) Cannot derive bet size using GMM CDF.")
    print("(d) Comparison cannot be fully made without GMM results.")

# --- Exercise 10.5 ---
print("\n\n--- Exercise 10.5 ---")
print("Repeat exercise 1, where you discretize m with a stepSize=.01, stepsize=.05, and stepSize=.1.")
# Exercise 1 was: "Using the formulation in Section10.3, plot the bet size (m) as a
# function of the maximum predicted probability (̃p) when ‖X‖= 2,3,…,10."
# The `discreteSignal` function is what we need here.

def discrete_bet_size(m_continuous, step_size):
    discretized = (m_continuous / step_size).round() * step_size
    discretized = np.clip(discretized, -1, 1) # Cap and floor
    return discretized

p_tilde_plot_values = np.linspace(0.01, 0.99, 200)
step_sizes_10_5 = [0.01, 0.05, 0.1]

for step_size in step_sizes_10_5:
    plt.figure(figsize=(12, 8))
    for num_classes in range(2, 11):
        continuous_bet_sizes = []
        # Generate p_tilde values appropriate for this num_classes
        current_p_tilde_range = [p for p in p_tilde_plot_values if p > (1.0/num_classes) + 1e-6]
        if not current_p_tilde_range and p_tilde_plot_values[0] <= (1.0/num_classes):
            current_p_tilde_range = np.linspace((1.0/num_classes) + 1e-6, 0.999, 100)
        
        if not isinstance(current_p_tilde_range, list): # Ensure it's iterable for the loop
            current_p_tilde_range_list = current_p_tilde_range.tolist()
        else:
            current_p_tilde_range_list = current_p_tilde_range

        for p_t in current_p_tilde_range_list:
            m = calculate_bet_size_m(p_t, num_classes)
            continuous_bet_sizes.append(m)
        
        if current_p_tilde_range_list and continuous_bet_sizes:
            discretized_m = discrete_bet_size(np.array(continuous_bet_sizes), step_size)
            plt.plot(current_p_tilde_range_list, discretized_m, label=f'||X|| = {num_classes}')

    plt.xlabel('Maximum Predicted Probability (p̃)')
    plt.ylabel(f'Discretized Bet Size (m) (step={step_size})')
    plt.title(f'Discretized Bet Size vs. p̃ (Step Size = {step_size})')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()