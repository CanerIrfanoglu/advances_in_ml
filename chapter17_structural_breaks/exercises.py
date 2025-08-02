# Chat LINK: https://aistudio.google.com/app/prompts?state=%7B%22ids%22%3A%5B%221oCd8f2TXSIV2HVvZaNUqrB7ZJigPOPxf%22%5D%2C%22action%22%3A%22open%22%2C%22userId%22%3A%22111971824642554314977%22%2C%22resourceKeys%22%3A%7B%7D%7D&usp=drive_link
# Nice summary table at the end!

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

df = pd.read_csv('./data/dollar_df_2025_14_days.csv', index_col=0)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# 17.1(a) Apply the Brown-Durbin-Evans method. Does it recognize the dot-com
# bubble?
# (We will use 2 weeks of bitcoin data instead where there is ~ 5 days sideways action then a break taking 2 days then side ways)

# This test checks if a predictive model has broken. We'll use a simple model: 
# predicting a bar's return from the previous bar's return 
# (Return_t = α + β * Return_{t-1}). 
# We'll see if the β (the momentum/mean-reversion factor) breaks down around May 8th.




# Step 1: Data Cleaning - Explicitly handle potential NaNs from the start
print(f"Original 'close' column contains {df['close'].isnull().sum()} NaN values.")
close_prices = df['close'].dropna()
print(f"Working with {len(close_prices)} non-NaN price points.")

# Step 2: Create returns FROM THE CLEAN DATA
# We will handle the NaNs from diff() and shift() all at once later.
log_price = np.log(close_prices)
returns = log_price.diff()

# Step 3: Define y and X from the same 'returns' series
# y is today's return
y = returns
# X is yesterday's return (plus a constant)
X = sm.add_constant(returns.shift(1))

# Step 4: Align and Drop NaNs Simultaneously (This is the most robust method)
# Combine y and X into a single dataframe
combined = pd.concat([y, X], axis=1)
combined.columns = ['y', 'const', 'X_t-1'] # Name columns for clarity

# Now, drop any row that has a NaN in EITHER column.
# This handles the NaNs from diff() and shift() in one go and ensures perfect alignment.
combined = combined.dropna()

solve_1a = False
if solve_1a:
    print("\nRunning Brown-Durbin-Evans Test...")

    # Step 5: Separate them back out. They are now guaranteed to be aligned.
    y = combined['y']
    X = combined[['const', 'X_t-1']]
    assert all(y.index == X.index)
    print(f"Data successfully aligned. Shape of y: {y.shape}, Shape of X: {X.shape}")

    # 2. Recursive estimation and CUSUM of errors
    k = 1000  # Minimum window size to initialize the model
    T = len(y)
    w_hats = [] # To store standardized recursive residuals

    for t in range(k, T):
        y_window, X_window = y.iloc[:t], X.iloc[:t]
        model = sm.OLS(y_window, X_window).fit()
        
        # Get the next observation to predict
        X_next = X.iloc[t:t+1]
        
        # Prediction error (raw residual)
        pred_error = y.iloc[t] - model.predict(X_next)[0]
        
        # Variance of the prediction error
        pred_var = model.mse_resid * (1 + (X_next.values @ np.linalg.inv(X_window.T @ X_window) @ X_next.values.T)[0,0])
        
        # Standardized residual
        if pred_var > 0:
            w_hat = pred_error / np.sqrt(pred_var)
            w_hats.append(w_hat)

    # Create a series of standardized residuals
    w_hat_series = pd.Series(w_hats, index=y.index[k:])

    # 3. Calculate the CUSUM statistic
    # We normalize by the overall std dev of the standardized errors
    cusum_stat = w_hat_series.cumsum() / w_hat_series.std()

    # 4. Visualization
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot BDE CUSUM statistic
    ax1.plot(cusum_stat, color='r', label='BDE CUSUM Statistic')
    ax1.axhline(0, color='black', linestyle='--', lw=1)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('CUSUM Statistic', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_title('BDE CUSUM Test for Structural Break vs. BTC Price')

    # Plot BTC price on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df.loc[cusum_stat.index, 'close'], label='BTC Price (right axis)', alpha=0.6)
    ax2.set_ylabel('BTC Price (USD)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Add a vertical line for the suspected break
    break_date = '2025-05-08'
    ax1.axvline(pd.to_datetime(break_date), color='purple', linestyle=':', lw=2, label=f'Suspected Break ({break_date})')

    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    plt.show()


# QUESTION: The BDE plot seems to be lagging though. How it is more valuable than say MACD which is also lagging

# You are right: the BDE CUSUM is a lagging indicator. It cannot, by definition, move before the price does. 
# However, its value is immense and fundamentally different from the MACD's.
# Here's why it is more valuable, broken down into key advantages.
# 1. It Measures a Deeper Concept: Model Failure vs. Price Momentum
# This is the most important distinction.
# MACD: Measures the momentum of price. It's a "dumb" calculation based purely on moving averages of the price itself. It answers the question: 
# "Is the price going up or down, and how fast?"
# BDE CUSUM: Measures the failure of a model. It's a second-order, "intelligent" calculation. 
# It answers a much more profound question: "Are the underlying rules of the market, which I have tried to model, still holding true?"
# Why this is more valuable:
# The MACD can be positive in a random, choppy market that happens to drift up. 
# The BDE CUSUM will only become strongly positive if there is a systematic, persistent, and surprising trend that consistently breaks the expectations of a rational model. It filters out random drift and isolates periods of true market hysteria or panic. It's measuring the character of the move, not just the move itself.
# 2. It is Model-Driven and Adaptable (Your "Secret Sauce")
# MACD: Has fixed parameters (e.g., 12, 26, 9). Everyone sees the same MACD. There is no proprietary edge.
# BDE CUSUM: The output is entirely dependent on the model you choose to test. This is where your edge comes from.
# Our simple model was Return = f(Last_Return).
# A more sophisticated model could be: Return = f(Last_Return, Volatility, Order_Flow_Imbalance).
# Now, the CUSUM isn't just detecting a trend; it's detecting when a trend is happening that cannot be explained by volatility or order flow. 
# This is an incredibly powerful piece of information. You can build a CUSUM that is unique to your specific view of the market.
# 3. It's a Measure of "Surprise" (Information Content)
# MACD: A positive crossover is a positive crossover. It doesn't tell you how surprising or significant that crossover is in the current context.
# BDE CUSUM: The entire statistic is built on standardized prediction errors (ŵ_t). It's not just summing the errors; it's summing the surprise content of each error. A large price move on a volatile day is less surprising (and contributes less to the CUSUM) than a medium price move on a quiet day. It automatically adjusts for the context of volatility, giving more weight to moves that truly signal new information.
# 4. It Provides Clearer Regime Boundaries
# MACD: The zero line is a weak separator of bullish/bearish territory. The MACD can whipsaw across the zero line frequently in a sideways market, generating many false signals.
# BDE CUSUM: As your chart beautifully shows, the zero line is a much more robust "line in the sand." The CUSUM tends to stay on one side of zero for a prolonged period, defining the entire regime. 
# When it does cross, it's a major, significant event. 
# The CUSUM from your chart generated exactly two major regime-change signals 
# (the V-bottom reversal and the subsequent peak), whereas a MACD would have likely 
# generated dozens of smaller, less meaningful signals in the same period.
# Summary: Lagging But Smarter
# Think of it like this:
# MACD is a smoke detector. It beeps when it sees smoke. It could be burnt toast or a house fire. It's fast but often wrong about the significance.
# BDE CUSUM is a seismograph. It registers tremors. It lags the start of the earthquake by a few seconds, but it can tell you the difference between a truck driving by and a genuine magnitude 7.0 event. 
# It also tells you where the fault line that broke is located (i.e., which part of your model failed).
# So, while both are lagging, the BDE CUSUM provides a vastly richer, more robust, and more customizable signal. 
# You are trading based on a fundamental shift in the market's statistical properties, 
# not just on a simple moving average of price. That is a significant and tradable edge.

# (b) ApplytheChu-Stinchcombe-Whitemethod.Doesitfindabubblein2007–
# 2008?
# NOTE THIS TAKES LONG LIKE 30 MINS

solve_1b = False
if solve_1b:
    print("\nRunning Chu-Stinchcombe-White Test... (This may ~30 mins)")

    # 1. Set parameters
    log_price = np.log(df['close'])
    window_size = 2000 # Test for breaks within a rolling window of 2000 bars
    csw_stats = []
    dates = []

    # 2. Rolling window implementation
    # We iterate through the data, and for each point, we look back `window_size` bars
    # and test for a break within that lookback period.
    for t in range(window_size, len(log_price)):
        log_price_window = log_price.iloc[t-window_size:t]
        
        s_nt_scores = []
        # Test every possible start date `n` within the window
        for n in range(len(log_price_window) - 20): # -20 to ensure min length for std dev
            y_t = log_price_window.iloc[-1]
            y_n = log_price_window.iloc[n]
            
            # Volatility of the sub-period
            sigma_hat = log_price_window.iloc[n:].diff().std()
            
            duration = len(log_price_window) - n
            
            if sigma_hat > 0 and duration > 1:
                s_nt = (y_t - y_n) / (sigma_hat * np.sqrt(duration))
                s_nt_scores.append(s_nt)
                
        # The statistic for time `t` is the supremum (max) of all scores
        if s_nt_scores:
            csw_stats.append(np.max(s_nt_scores))
            dates.append(log_price_window.index[-1])

    csw_series = pd.Series(csw_stats, index=dates)

    # 3. Visualization
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot CSW statistic
    ax1.plot(csw_series, color='g', label='CSW Statistic')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('CSW Statistic (Supremum Score)', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_title('CSW "Bubble" Test vs. BTC Price')

    # Add a critical value line (e.g., 99th percentile of observed values)
    critical_value = csw_series.quantile(0.99)
    ax1.axhline(critical_value, color='orange', linestyle='--', lw=2, label=f'99% Critical Value ({critical_value:.2f})')

    # Plot BTC price on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df.loc[csw_series.index, 'close'], label='BTC Price (right axis)', alpha=0.5)
    ax2.set_ylabel('BTC Price (USD)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Add a vertical line for the suspected break
    break_date = '2025-05-08'
    ax1.axvline(pd.to_datetime(break_date), color='purple', linestyle=':', lw=2, label=f'Suspected Break ({break_date})')

    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    plt.show()

# INTERPRETATION OF CSW
# This plot is fascinating and shows a different side of the market than the BDE plot.
# Executive Summary
# Yes, the CSW test clearly identifies the rally on May 8th as a "bubble" event. It does this by showing a sharp, definitive spike in the test statistic that crosses a pre-defined critical value, signaling that the price action is no longer random noise.
# Detailed Interpretation
# The Green Line (CSW Statistic): This line represents the "bubble score" at any given time. It's the result of the "fastest sprint" competition within the last 2000 bars. It is naturally very spiky and volatile because the "fastest sprint" can change dramatically from one moment to the next.
# The Orange Line (99% Critical Value): This is your statistical significance threshold. Think of it as the "Olympic qualifying time." Any sprint that is faster than this time (i.e., any green spike that goes above this orange line) is considered a statistically significant, non-random event—a "bubble."
# Phase 1: The Pre-Bubble Period (May 3 - May 7)
# We see the green line is very choppy. It has many spikes, but none of them manage to decisively cross and stay above the orange critical value line.
# Interpretation: The test is detecting many small, short-lived trends (both up and down), which is typical of a noisy, choppy market. However, it correctly concludes that none of these trends are sustained or powerful enough to be classified as a true bubble. The market is just being noisy.
# Phase 2: The Bubble Detection (The Spike on May 8)
# Right around the end of May 7 and the start of May 8, we see a massive green spike that clearly pierces the orange critical value line. It reaches a peak of over 4.0, well above the 3.57 threshold.
# Interpretation: This is the "Aha!" moment. The test's radar has picked up a definitive signal. The code found a start date (n) at the price low near May 7 and an end date (t) during the powerful rally. The resulting "surprise score" was so high that it crossed the significance threshold. The test is officially declaring that a bubble-like explosive trend has begun.
# Phase 3: The Post-Bubble Period (After May 9)
# The green statistic drops back below the critical value, though it remains highly elevated and volatile.
# Interpretation: The initial, single, most powerful explosive trend is over. The market has now transitioned into a new phase of high volatility with many strong up and down moves, but not a single, sustained bubble trend like the one on May 8th.
# BDE vs. CSW: A Tale of Two Detectives
# Your BDE plot was a smooth, cumulative story of a slow-witted detective realizing their entire worldview was wrong. It was great for defining the entire regime.
# This CSW plot is the work of a hyperactive detective with a radar gun, constantly looking for the single fastest car. It's noisy, but it's excellent at pinpointing the exact moment the speed limit was broken.
# A trading signal from this chart would be generated at the moment the green line crosses above the orange line. This would have been a perfect entry signal for the May 8th rally.


# 17.2 Onadollar bar serieson E-mini S&P 500 futures,
# THIS ONE CREATES CONTINUOUS T-SCORES assuming the market is initially choppy and non-trending.
# i.e. t=0. As t increases it confirms the magnitude of a trend. Therefore, it's a lagging indicator but
# it is also dangerous trying to use it for predicting a trend reversal.

# (a) Compute the SDFC (Chow-type) explosiveness test.What break date does
# this method select? Is this what you expected?
# --- Part (a): SDFC (Chow-type) Test ---
from tqdm import tqdm
solve_2a = False
if solve_2a:
    
    print("\n(a) Running SDFC (Chow-type) Test...")

    # Prepare data for the regression: Δy_t = δ*y_{t-1}*D_t + ε_t
    price_diff = log_price.diff().dropna()
    price_lag = log_price.shift(1).dropna()
    aligned_df = pd.concat([price_diff, price_lag], axis=1).dropna()
    aligned_df.columns = ['price_diff', 'price_lag']

    # Set parameters
    n_obs = len(aligned_df)
    trim = 0.15 # Trim 15% from the beginning and end for break date search
    min_break_idx = int(n_obs * trim)
    max_break_idx = int(n_obs * (1 - trim))

    dfc_scores = {} # To store DFC score for each potential break date

    # Loop through all possible break dates
    for tau_idx in tqdm(range(min_break_idx, max_break_idx), desc="SDFC Calculation"):
        # Create the dummy variable D_t[τ*]
        dummy = np.zeros(n_obs)
        dummy[tau_idx:] = 1 # 0 before break, 1 after

        # Create the regressor y_{t-1} * D_t
        X = (aligned_df['price_lag'] * dummy).values.reshape(-1, 1)
        y = aligned_df['price_diff'].values

        # Fit OLS model: Δy = δ * (y_lag * Dummy)
        model = sm.OLS(y, X).fit()
        
        # Get the t-statistic for our δ coefficient
        t_stat = model.tvalues[0]
        
        break_date = aligned_df.index[tau_idx]
        dfc_scores[break_date] = t_stat

    # Convert results to a pandas Series
    dfc_series = pd.Series(dfc_scores)

    # Find the SDFC (the max DFC score) and the corresponding break date
    sdfc_score = dfc_series.max()
    selected_break_date = dfc_series.idxmax()

    print(f"\nSDFC Test Results:")
    print(f"  > The maximum DFC score (SDFC) is: {sdfc_score:.4f}")
    print(f"  > The selected break date is: {selected_break_date}")

    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.set_title('Chow-Type DFC Scores by Potential Break Date vs. Price')

    # Plot the DFC scores on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Potential Break Date (τ*)')
    ax1.set_ylabel('DFC t-statistic', color=color)
    ax1.plot(dfc_series, color=color, label='DFC Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(False) # Turn off grid for the first axis if desired

    # Add a vertical line for the selected break date
    ax1.axvline(selected_break_date, color='r', linestyle='--', 
                label=f'Selected Break: {selected_break_date.date()}')

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()  

    # Plot the BTC price on the secondary y-axis
    color = 'tab:gray'
    ax2.set_ylabel('BTC Price (USD)', color=color)  
    # We plot the price series aligned to the dates where we have a DFC score
    ax2.plot(df.loc[dfc_series.index, 'close'], color=color, alpha=0.6, label='BTC Price (right axis)')
    ax2.tick_params(axis='y', labelcolor=color)

    # Combine legends from both axes
    # This is a bit more robust for twinx plots
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Ensure the plot layout is tight
    fig.tight_layout()  
    plt.show()



# (b) ComputeandplottheSADFvaluesforthisseries.Doyouobserveextreme
# spikesaroundthedot-combubbleandbeforetheGreatRecession?Didthe
# burstsalsocause spikes?

# --- Part (b): SADF Test ---
solve_2b = True
if solve_2b:

    print("\n(b) Running SADF Test...")
    from statsmodels.tsa.stattools import adfuller

    # Set parameters
    min_window = 1000  # Start calculating SADF after 1000 observations
    sadf_values = {}

    # Loop through the data on an expanding window
    for end_point in tqdm(range(min_window, len(log_price)), desc="SADF Calculation"):
        window = log_price.iloc[:end_point]
        
        # Run the ADF test on the current window.
        # We are interested in the t-statistic.
        # 'c' means we include a constant (drift) in the regression.
        adf_result = adfuller(window, regression='c', autolag=None)
        sadf_values[window.index[-1]] = adf_result[0] # The t-statistic

    sadf_series = pd.Series(sadf_values)

    # Visualization
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot SADF statistic
    ax1.plot(sadf_series, color='purple', label='SADF Statistic')
    ax1.axhline(0, color='black', linestyle='--', lw=1)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('SADF t-statistic', color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.set_title('SADF Test Statistic vs. BTC Price')

    # Plot BTC price on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df.loc[sadf_series.index, 'close'], label='BTC Price (right axis)', alpha=0.5)
    ax2.set_ylabel('BTC Price (USD)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Add a vertical line for the identified break
    # ax1.axvline(selected_break_date, color='r', linestyle=':', lw=2, label=f'Identified Break ({selected_break_date.date()})')

    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    plt.show()


# Exercise 17.3: Identifying Market Conditions
# (a) Determine the periods where the series exhibited (i) Steady, (ii) Unit-Root, and (iii) Explosive conditions.
# This question is answered perfectly by interpreting our SADF plot. The SADF t-statistic is a direct measure of these conditions.
# (i) Steady Conditions (Mean-Reverting):
# Theory: A "steady" or mean-reverting process has a strong pull back to an average. The ADF test is very good at rejecting the random walk hypothesis for these, resulting in a large negative t-statistic.
# In Your Data: The periods from May 3rd to May 5th and again from May 13th onwards. In these periods, the purple SADF line is at its lowest levels (around -2.5 to -3.0), indicating strong evidence against a random walk and in favor of a more predictable, grinding, or mean-reverting behavior.
# (ii) Unit-Root Conditions (Random Walk):
# Theory: This is the baseline "unpredictable" state. The ADF test statistic will be a small negative number, typically between -2.5 and 0, where we cannot confidently reject the random walk hypothesis.
# In Your Data: The period from roughly May 9th to May 13th. After the explosive peak, the SADF statistic falls into this intermediate zone. The market is no longer in an explosive bubble, but has settled into a more "normal" state of choppy, unpredictable movement.
# (iii) Explosive Conditions (Bubble):
# Theory: A bubble is the opposite of a mean-reverting process. It pushes the ADF statistic upwards towards and past zero.
# In Your Data: The period from late May 7th to May 9th. The SADF statistic makes a dramatic surge from its lows, crosses zero, and peaks above +1. This is the unmistakable signature of an explosive, non-random, trending market—a bubble.
# (b) Compute QADF (Quantile ADF).
# Concept: The SDFC test we ran gave us a single "break date" by picking the maximum (the supremum) of all the test scores. This can be sensitive to a single outlier. QADF is a more robust alternative.
# How it would work: Instead of taking the absolute maximum score, the QADF method would calculate the 95th percentile of all the DFC scores from your plot.
# Expected Result: The result would be very similar. The peak of the 95th percentile scores would still almost certainly occur around May 6th, but the resulting line would be smoother and less spiky than a plot of the pure maximum. It provides a more stable, "consensus" view of the break.
# (c) Compute CADF (Conditional ADF).
# Concept: This is another, even more robust alternative to the SDFC's maximum score.
# How it would work: It would first identify the top 5% of all DFC scores. Then, instead of picking one, it would compute the average of that entire top group.
# Expected Result: This would also select a break date around May 6th. Its key advantage is that it also gives you the standard deviation of the top scores. A small standard deviation would tell you that many different potential break dates around May 6th were all giving high scores, dramatically increasing your confidence that the signal is real and not just a random fluke.
# Exercise 17.4: Sub- and Super-Martingale Tests (SMT)
# These tests look for trends against time itself (like a rocket trajectory) rather than self-referential momentum. φ is a penalty against long trends to make the test fairer.
# (a) SMT for SM-Poly1 and SM-Poly2 (φ=1). What is their correlation?
# SM-Poly1: Fits a quadratic curve to the raw price: Price ≈ β * time²
# SM-Poly2: Fits a quadratic curve to the log price: Log(Price) ≈ β * time²
# Expected Result: Both tests would detect a powerful, accelerating trend during the rally on May 8th. Since log-price is a simple transformation of price, a curve in one is also a curve in the other. Their SMT scores would be very highly correlated.
# (b) SMT for SM-Exp (φ=1 vs φ=0.5). What is their correlation?
# Concept: We are running the same exponential trend test but with two different penalties.
# φ=1: Heavy penalty. Biased towards finding short, sharp trends.
# φ=0.5: Moderate penalty. A more balanced approach.
# Expected Result: Both would spike during the May 8th rally. Since they are derived from the same base model, their scores will be highly correlated. The φ=1 series would likely be spikier and more reactive to short-term bursts, while the φ=0.5 series might be slightly smoother.
# (c) SMT for SM-Power (φ=1 vs φ=0.5). What is their correlation?
# Concept & Result: The logic is identical to part (b). We are just changing the assumed trend shape from exponential to a power law. The two versions with different φ penalties will still be highly correlated as they are measuring the same underlying phenomenon.
# Exercise 17.5: Reciprocal Prices and Identifying Bursts (Crashes)
# (a) Is this transformation (1/price) needed to identify bursts?
# No, it is not strictly needed. The tests in this chapter are robust enough to detect crashes ("bursts") directly. A crash is simply a "negative bubble"—a strong, persistent, non-random trend in the downward direction.
# Why it's not needed: Most tests are based on estimating a coefficient (β or δ) and its t-statistic. A crash will simply result in a large negative coefficient with a large negative t-statistic. The math is sign-agnostic. The BDE CUSUM test is also inherently bidirectional, as a crash would cause a cascade of negative errors, making the CUSUM plummet.
# (b) What methods in this chapter can identify bursts without this transformation?
# Virtually all of them.
# BDE CUSUM: Yes. A crash causes systematic over-prediction, leading to a plunging CUSUM line.
# CSW Test: Yes. The core term (y_t - y_n) becomes a large negative number. The test can be set up to look for large absolute deviations.
# SDFC (Chow-type): Yes. It can detect a significant negative δ, indicating a switch to a crash regime. Our DFC plot showed scores becoming negative during the big sell-off after May 9th.
# SADF: Yes. A crash is a strong departure from a random walk. It would cause a dramatic change in the t-statistic (likely moving it to a large negative value, as seen on May 3-4 in your plot).
# SMT: Yes. The trend models (e.g., y ≈ βt²) can perfectly capture a crash if the trend coefficient β is found to be significantly negative.
