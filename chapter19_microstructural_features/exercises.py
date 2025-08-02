# Chat LINK: https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%2213enbE1Az-FO6sniPHvokbfJnAc9un_wX%22%5D,%22action%22:%22open%22,%22userId%22:%22111971824642554314977%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing, https://drive.google.com/file/d/1SNyH1GBRogacPv80DlXJKVwZOUmbZ2pZ/view?usp=sharing, https://drive.google.com/file/d/1d9tVeLoIDUuQ7YggjoJjw6eC18CNuxie/view?usp=sharing, https://drive.google.com/file/d/1jduOmgMmNmE4cmNmIi7VWOih3CrYiQeh/view?usp=sharing, https://drive.google.com/file/d/1xvdGl26eIIKGvDr5dATGdbebbj_GpXUM/view?usp=sharing
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

btc_data_raw = pd.read_csv('./data/BTCUSDT_combined_may2025_1_to_14.csv')
btc_data_1day = pd.read_csv('./data/BTCUSDT_2025-05-01.csv')
btc_dollar_df = pd.read_csv('./data/dollar_df_2025_14_days.csv', index_col=0)
btc_dollar_df['timestamp'] = pd.to_datetime(btc_dollar_df['timestamp'])
dollar_df_rich = pd.read_csv('./data/dollar_df_2025_14_days_w_micro.csv')
dollar_df_rich['timestamp'] = pd.to_datetime(dollar_df_rich['timestamp'])

# btc_dollar_df.set_index('timestamp', inplace=True)


print('Shape 14 days =', btc_data_raw.shape)
print('Shape 1 day =', btc_data_1day.shape)

solve_1a = False
if solve_1a:
    # 19.1 (a) Apply the tick rule to derive the series of trade signs.
    # df = btc_data_raw.copy()
    df = btc_data_1day.copy()

    # --- Step 1: Apply the Tick Rule ---

    # The tick rule logic:
    # b_t = 1   if Δp_t > 0
    # b_t = -1  if Δp_t < 0
    # b_t = b_{t-1} if Δp_t = 0
    # b_0 is arbitrarily set to 1.

    # Calculate the price change from the previous trade
    price_diff = df['price'].diff()

    # Get the sign of the price change (+1, -1, or 0)
    # np.sign() is perfect for this.
    tick_rule_signs = np.sign(price_diff)

    # Handle the zero-tick case (Δp_t = 0) by carrying forward the last known sign.
    # We replace 0s with NaN, then use forward-fill.
    tick_rule_signs = tick_rule_signs.replace(0, np.nan).ffill()

    # Handle the very first trade. The first diff is NaN.
    # As per the text, "b_0 is arbitrarily set to 1", so we use 1 as the initial sign.
    tick_rule_signs = tick_rule_signs.fillna(1)

    # Add the calculated signs to our DataFrame
    df['tick_rule_sign'] = tick_rule_signs.astype(int)


    # --- Step 2: Prepare the Actual Labels ---

    # Convert the 'side' column to numeric labels for comparison
    # 'buy' -> 1, 'sell' -> -1
    df['actual_sign'] = df['side'].apply(lambda x: 1 if x == 'buy' else -1)


    # --- Step 3: Compare the Results ---

    # Calculate and print the overall accuracy
    accuracy = accuracy_score(df['actual_sign'], df['tick_rule_sign'])
    print(f"Overall Accuracy of the Tick Rule: {accuracy:.2%}")
    print("\n")

    # For a more detailed breakdown, we can use a classification report
    print("--- Detailed Classification Report ---")
    # Note: scikit-learn's report expects string labels for clarity
    report = classification_report(
        df['actual_sign'].astype(str),
        df['tick_rule_sign'].astype(str),
        target_names=['Sell (-1)', 'Buy (1)']
    )
    print(report)

    # We can also view the confusion matrix
    print("--- Confusion Matrix ---")
    # Rows = Actual, Columns = Predicted
    cm = confusion_matrix(df['actual_sign'], df['tick_rule_sign'], labels=[-1, 1])
    print("          Predicted")
    print("          Sell  Buy")
    print(f"Actual Sell  {cm[0][0]}    {cm[0][1]}")
    print(f"       Buy   {cm[1][0]}    {cm[1][1]}")


solve_19_2 = False
# Roll model is applicable only for orderbook level data

if solve_19_2:
    # df = btc_dollar_df.copy()
    df = btc_data_1day.copy()
    # 19.2 Compute the Roll model on the time series of E-mini S&P 500 futures tick data
    # (a) What are the estimated values of 𝜎2 and c?
    # 1. Calculate the series of price changes (Δp_t)
    price_changes = df['price'].diff().dropna() # drop the first NaN value

    # 2. Calculate the first-order autocovariance of price changes
    # This is the key input for the Roll model
    autocovariance = price_changes.autocorr(lag=1)
    print(f"First-order autocovariance of price changes: {autocovariance:.6f}")

    # 3. Check if the model is applicable (autocovariance must be negative)
    if autocovariance >= 0:
        print("\nWARNING: Autocovariance is non-negative. The Roll model's assumptions are violated.")
        print("This indicates price momentum, not bid-ask bounce, dominates the series.")
        c = np.nan
        sigma_u_sq = np.nan
    else:
        # 4. Calculate 'c', the effective half-spread
        c = np.sqrt(-autocovariance)
        
        # 5. Calculate the variance of price changes
        variance_p = price_changes.var()
        
        # 6. Calculate 'σ_u^2', the variance of fundamental value changes
        sigma_u_sq = variance_p - (2 * c**2)

    # Display the final results for part (a)
    print("\n--- (a) Estimated Roll Model Values ---")
    print(f"Effective half-spread (c): {c:.6f}")
    print(f"Fundamental value variance (σ_u^2): {sigma_u_sq:.6f}")





#19.3 c)
import matplotlib.pyplot as plt
solve_19_3c = False
if solve_19_3c:
    dollar_df = btc_dollar_df.copy()

    # Ensure we have enough data
    if len(dollar_df) < 2:
        print("Not enough data to compute volatility. Need at least 2 bars.")
    else:
        # --- Step 1: Calculate Volatility Features (same as before) ---
        window_size = 50
        
        # High-Low Volatility (Parkinson)
        valid_hl = (dollar_df['high'] > 0) & (dollar_df['low'] > 0) & (dollar_df['high'] >= dollar_df['low'])
        dollar_df['vol_hl'] = np.nan
        dollar_df.loc[valid_hl, 'vol_hl'] = (1 / np.sqrt(4 * np.log(2))) * np.log(dollar_df.loc[valid_hl, 'high'] / dollar_df.loc[valid_hl, 'low'])
        
        # Smoothed High-Low Volatility
        dollar_df['vol_hl_smooth'] = dollar_df['vol_hl'].rolling(window=window_size).mean()

        # Close-to-Close Volatility
        log_returns = np.log(dollar_df['close'] / dollar_df['close'].shift(1))
        dollar_df['vol_c2c'] = log_returns.rolling(window=window_size, min_periods=2).std()

        # Volatility Ratio
        epsilon = 1e-12 
        dollar_df['vol_ratio'] = dollar_df['vol_c2c'] / (dollar_df['vol_hl_smooth'] + epsilon)


        # --- Step 2: Create the Stacked Subplot Visualization ---
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create a figure with two subplots, stacked vertically.
        # `sharex=True` links their x-axes.
        # `gridspec_kw` makes the bottom plot shorter than the top one.
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, 
            ncols=1, 
            figsize=(18, 10), 
            sharex=True, 
            gridspec_kw={'height_ratios': [3, 1]}
        )
        
        # --- Top Subplot (ax1): Price and Volatility Estimators ---
        
        # Use a dual-axis for the top plot as well
        ax1_twin = ax1.twinx()
        
        # Plot volatility on the left axis of the top plot
        ax1.set_ylabel('Volatility', fontsize=12)
        p1, = ax1.plot(dollar_df['timestamp'], dollar_df['vol_hl_smooth'], color='green', linewidth=2, label=f'Smoothed HL Vol ({window_size}-bar mean)')
        p2, = ax1.plot(dollar_df['timestamp'], dollar_df['vol_c2c'], color='red', linewidth=2, label=f'C2C Vol ({window_size}-bar std)')
        ax1.tick_params(axis='y')
        
        # Plot price on the right axis of the top plot
        ax1_twin.set_ylabel('BTC Price', fontsize=12)
        p3, = ax1_twin.plot(dollar_df['timestamp'], dollar_df['close'], color='black', alpha=0.6, linewidth=1.5, label='BTC Close Price')
        ax1_twin.tick_params(axis='y')

        ax1.set_title('Price vs. Smoothed Volatility Estimators', fontsize=16)
        
        # Add a unified legend for the top plot
        plots = [p1, p2, p3]
        ax1.legend(plots, [p.get_label() for p in plots], loc='upper left')

        
        # --- Bottom Subplot (ax2): Volatility Ratio ---
        ax2.set_ylabel('Volatility Ratio', fontsize=12)
        ax2.plot(dollar_df['timestamp'], dollar_df['vol_ratio'], color='darkorange', linewidth=2, label='Ratio (C2C / HL)')
        
        # Add the crucial horizontal line at 1.0
        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Balance Point (Ratio=1.0)')
        
        ax2.set_xlabel('Timestamp', fontsize=14)
        ax2.legend(loc='upper left')
        
        # Set a reasonable y-limit for the ratio plot to handle extreme outliers
        # For example, cap it at 4 for better visualization
        q99 = dollar_df['vol_ratio'].quantile(0.99)
        ax2.set_ylim(0, q99 * 2) 
        

        # Final Touches
        fig.suptitle('Comprehensive Market Dynamics: Price, Volatility, and Structure', fontsize=20, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for the main title
        
        plt.show()

solve_19_5 = True
if solve_19_5:


    print("--- Solving Exercise 19.5(b): Kyle's Lambda ---")

    # 1. Prepare the variables for regression
    y = dollar_df_rich['close'].diff().dropna()
    # The independent variable is the Order Flow Imbalance (OFI)
    X = dollar_df_rich['ofi'].loc[y.index] # Ensure X and y are aligned

    # Add a constant (intercept) to the model, which is good practice
    X = sm.add_constant(X)

    # 2. Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(y, X).fit()

    # 3. Extract Kyle's Lambda
    # Lambda is the coefficient of our 'ofi' variable
    kyle_lambda = model.params['ofi']

    # Print the results
    print(f"Estimated Kyle's Lambda (λ): {kyle_lambda:.6f}")
    print("\nFull OLS Regression Summary:")
    print(model.summary())

solve_19_7 = True
if solve_19_7:

    # Meaning: Amihud's lambda is a tiny number by nature. A value like 2.5e-09 means 
    # that for every $1 million traded, the price is expected to move by 2.5e-09 * 1,000,000 = 0.0025, or 0.25%.
    # Consistency with Kyle's Lambda: Both lambdas are measures of illiquidity/price 
    # impact. While their units and scales are different, they should both tell a similar story. 
    # If we were to calculate these on a rolling basis, 
    # we would expect periods where Kyle's Lambda is high (high price impact) to also be periods where Amihud's Lambda is high. 
    # Both should be positively correlated with volatility.

    # Assuming 'dollar_df_rich' also has 'VWAP' and 'volume' columns
    # --- Placeholder Update ---
    # dollar_df_rich['VWAP'] = dollar_df_rich['close']
    # dollar_df_rich['volume'] = 10.6 # Approx volume for $1M at $94k price
    # --- End Placeholder Update ---

    print("\n--- Solving Exercise 19.7: Amihud's Lambda ---")

    # 1. Prepare variables
    # Dependent variable: absolute log return
    y_amihud = np.log(dollar_df_rich['close'] / dollar_df_rich['close'].shift(1)).abs().dropna()

    # Independent variable: Dollar Volume for the bar
    dollar_volume = (dollar_df_rich['VWAP'] * dollar_df_rich['volume']).loc[y_amihud.index]
    X_amihud = sm.add_constant(dollar_volume)
    X_amihud.columns = ['const', 'dollar_volume'] # Rename for clarity

    # 2. Fit the OLS model
    model_amihud = sm.OLS(y_amihud, X_amihud).fit()

    # 3. Extract Amihud's Lambda
    amihud_lambda = model_amihud.params['dollar_volume']

    print(f"Estimated Amihud's Lambda (λ): {amihud_lambda:.12f}")
    print("\nFull OLS Regression Summary:")
    print(model_amihud.summary())


solve_19_8 = False
if solve_19_8:

    # Assuming 'dollar_df_rich' has 'volume' and 'avg_trade_sign'
    # --- Placeholder Update ---
    dollar_df_rich['avg_trade_sign'] = dollar_df_rich['ofi'] / dollar_df_rich['volume']
    # --- End Placeholder Update ---

    print("\n--- Solving Exercise 19.8: VPIN ---")

    # 1. Estimate Buy and Sell Volume within each bar
    buy_vol_frac = (dollar_df_rich['avg_trade_sign'] + 1) / 2
    dollar_df_rich['V_buy'] = buy_vol_frac * dollar_df_rich['volume']
    dollar_df_rich['V_sell'] = (1 - buy_vol_frac) * dollar_df_rich['volume']

    # 2. Calculate VPIN over a rolling window
    # The original paper uses volume buckets. We'll use a rolling window of bars, e.g., 50.
    window = 50
    order_flow_imbalance = np.abs(dollar_df_rich['V_buy'] - dollar_df_rich['V_sell'])
    total_volume_window = dollar_df_rich['volume'].rolling(window=window).sum()
    imbalance_window = order_flow_imbalance.rolling(window=window).sum()

    dollar_df_rich['vpin'] = imbalance_window / total_volume_window

    # 3. Plot the series of VPIN and prices (as in 19.8b)
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(18, 10), sharex=True, 
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Top plot: Price
    ax1.set_title('BTC Price', fontsize=16)
    ax1.plot(dollar_df_rich['timestamp'], dollar_df_rich['close'], color='black', alpha=0.8)
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Bottom plot: VPIN
    ax2.set_title('VPIN (Toxicity of Order Flow)', fontsize=16)
    ax2.plot(dollar_df_rich['timestamp'], dollar_df_rich['vpin'], color='purple')
    ax2.set_ylabel('VPIN')
    ax2.set_xlabel('Timestamp')
    ax2.grid(True)

    fig.tight_layout()
    plt.show()

    print("\nVPIN Series Computed and Plotted.")