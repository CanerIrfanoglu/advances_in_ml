import pandas as pd
import numpy as np

# Load your raw data
# Make sure the path is correct
btc_data_raw = pd.read_csv('./data/BTCUSDT_combined_may2025_1_to_14.csv')

# --- Step 1: Pre-computation on Raw Tick Data ---
print("Step 1: Pre-processing raw tick data...")
transformed_df = pd.DataFrame()
# Note: Keeping timestamps as datetime objects is better for pandas operations
transformed_df['timestamp'] = pd.to_datetime(btc_data_raw['timestamp'], unit='ms', utc=False) 
transformed_df['side'] = btc_data_raw["side"]
transformed_df['quantity'] = btc_data_raw["volume"]
transformed_df['price'] = btc_data_raw["price"]
transformed_df['actual_sign'] = np.where(transformed_df['side'] == 'buy', 1, -1)
print("Pre-processing complete.")

# # This is only needed when we don't have side in the raw data. See Chapter 19.1 a)
# # For it's calculation and comparison to actual signs.
# # Calculate tick rule sign (b_t)
# price_diff = transformed_df['price'].diff()
# tick_rule_signs = np.sign(price_diff)
# tick_rule_signs = tick_rule_signs.replace(0, np.nan).ffill() # Propagate for zero-ticks
# tick_rule_signs = tick_rule_signs.fillna(1).astype(int) # Set initial sign to 1

# # Add this crucial feature to our DataFrame
# transformed_df['tick_rule_sign'] = tick_rule_signs
# print("Tick Rule signs added to the DataFrame.\n")


# --- Step 2 & 3: Refactored and Fully Enhanced Aggregation Function ---
def aggregate_data(df, bar_type='dollar', threshold=1000000):
    """
        A DataFrame where each row represents an aggregated bar, indexed by
        timestamp. The columns include:

        --- Standard Bar Features ---
        - timestamp: The timestamp of the first trade in the bar.
        - open, high, low, close: Standard OHLC price values for the bar.
        - VWAP: Volume-Weighted Average Price for the bar.
        - volume: The total volume (e.g., number of BTC) traded in the bar.
        - num_ticks: The total number of individual trades within the bar.

        --- Order Flow Features ---
        - ofi: Order Flow Imbalance. The net signed volume (sum[sign * volume]).
               A positive value indicates net buying pressure; negative indicates
               net selling pressure.

        --- Advanced Microstructural Features

        - mean_trade_size: The average size of trades within the bar.
        - std_trade_size: The standard deviation of trade sizes, indicating trade size heterogeneity.
        - skew_trade_size: Skewness of the trade size distribution. 
            High positive skew suggests many small trades mixed with a few very large ones 
            (potential "iceberg" orders).
        - kurt_trade_size: Kurtosis ("fat tails") of the trade sizedistribution. 
            High kurtosis indicates a higher frequency of extreme-sized trades (both large and small).
        - max_trade_size: The size of the single largest trade in the bar.
            A direct flag for large, potentially uninformed, orders.
        - volume_concentration: The Herfindahl-Hirschman Index (HHI) of trade
            volumes. Measures the "lumpiness" of trading. A value near 1 means a few large trades dominated;
            a value near 0 means trading was highly fragmented.
        - front_loaded_vol_pct: Measures if volume was front-loaded in time within the bar's duration.
        - signed_vol_autocorr: First-order serial correlation of signed volumes.
            A high positive value is a strong indicator of "order splitting," where a large institutional
            order is being worked over many smaller trades.
    """
    
    # --- Helper function to calculate all features for a given slice of data ---
    def _calculate_bar_features(tick_data_slice):
        if tick_data_slice.empty:
            return None
        
        total_volume = tick_data_slice['quantity'].sum()
        if total_volume == 0: return None # Skip bars with zero volume

        # Standard OHLCV Features
        open_price = tick_data_slice['price'].iloc[0]
        high_price = tick_data_slice['price'].max()
        low_price = tick_data_slice['price'].min()
        close_price = tick_data_slice['price'].iloc[-1]
        vwap_value = (tick_data_slice['price'] * tick_data_slice['quantity']).sum() / total_volume
        num_ticks = len(tick_data_slice)
        
        # Order Flow Imbalance (OFI)
        ofi = (tick_data_slice['actual_sign'] * tick_data_slice['quantity']).sum()
        
        ### NEW ### --- EXERCISE 19.9: Order Size Distribution Moments ---
        mean_trade_size = tick_data_slice['quantity'].mean()
        std_trade_size = tick_data_slice['quantity'].std()
        skew_trade_size = tick_data_slice['quantity'].skew()
        kurt_trade_size = tick_data_slice['quantity'].kurt()
        max_trade_size = tick_data_slice['quantity'].max()

        ### NEW ### --- EXERCISE 19.11 (Adapted): Volume Concentration (HHI) ---
        volume_fractions = tick_data_slice['quantity'] / total_volume
        volume_concentration = (volume_fractions**2).sum()


        front_loaded_vol_pct = np.nan # Default to NaN
        if num_ticks > 1:
            start_time = tick_data_slice['timestamp'].iloc[0]
            end_time = tick_data_slice['timestamp'].iloc[-1]
            duration = end_time - start_time
            
            if duration.total_seconds() > 0:
                # Define cutoff as first 25% of the bar's life
                time_cutoff = start_time + 0.25 * duration
                
                # Get volume traded before the cutoff
                front_loaded_volume = tick_data_slice[tick_data_slice['timestamp'] <= time_cutoff]['quantity'].sum()
                
                front_loaded_vol_pct = front_loaded_volume / total_volume
            else:
                # If duration is zero (all trades have same timestamp), concentration is 100%
                front_loaded_vol_pct = 1.0

        ### NEW ### --- EXERCISE 19.12: Signed Volume Autocorrelation ---
        signed_vol_autocorr = np.nan
        if num_ticks > 1:
            signed_volumes = tick_data_slice['actual_sign'] * tick_data_slice['quantity']
            signed_vol_autocorr = signed_volumes.autocorr(lag=1)

        # Return all features as a dictionary
        return {
            'timestamp': tick_data_slice["timestamp"].iloc[0],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'VWAP': vwap_value,
            'volume': total_volume,
            'num_ticks': num_ticks,
            'ofi': round(ofi, 8),
            ### NEW ###
            'mean_trade_size': round(mean_trade_size,8),
            'std_trade_size': round(std_trade_size,8),
            'skew_trade_size': round(skew_trade_size,8),
            'kurt_trade_size': round(kurt_trade_size,8),
            'max_trade_size': round(max_trade_size,8),
            'volume_concentration': round(volume_concentration,8),
            'front_loaded_vol_pct': round(front_loaded_vol_pct,2),
            'signed_vol_autocorr': round(signed_vol_autocorr,8)
        }

    bars_data = []
    start_index = 0
    
    if bar_type == 'tick':
        num_bars = len(df) // threshold
        for i in range(num_bars):
            start_idx = i * threshold
            end_idx = (i + 1) * threshold
            tick_data = df.iloc[start_idx:end_idx]
            bar_features = _calculate_bar_features(tick_data)
            if bar_features: bars_data.append(bar_features)
    
    elif bar_type == 'volume':
        current_volume = 0
        for i in range(len(df)):
            current_volume += df['quantity'].iloc[i]
            if current_volume >= threshold:
                tick_data = df.iloc[start_index : i + 1]
                bar_features = _calculate_bar_features(tick_data)
                if bar_features: bars_data.append(bar_features)
                current_volume = 0
                start_index = i + 1
    
    elif bar_type == 'dollar':
        current_dollar = 0
        for i in range(len(df)):
            current_dollar += df['price'].iloc[i] * df['quantity'].iloc[i]
            if current_dollar >= threshold:
                tick_data = df.iloc[start_index : i + 1]
                bar_features = _calculate_bar_features(tick_data)
                if bar_features: bars_data.append(bar_features)
                current_dollar = 0
                start_index = i + 1

    if not bars_data:
        return pd.DataFrame()
        
    final_df = pd.DataFrame(bars_data)
    # Handle NaNs that can arise from calculations on bars with few ticks
    final_df.fillna({
        'std_trade_size': 0, 
        'skew_trade_size': 0, 
        'kurt_trade_size': 0, 
        'signed_vol_autocorr': 0
    }, inplace=True)
    
    return final_df


# --- Step 4: Run the function and save the results ---
print("Step 2: Aggregating data into dollar bars...")

# Using a threshold of $1,000,000 for the dollar bars as an example
DOLLAR_BAR_THRESHOLD = 1000000
dollar_df_rich = aggregate_data(transformed_df, "dollar", threshold=DOLLAR_BAR_THRESHOLD)

print("\n--- Aggregated Dollar Bars with All New Features ---")
# Displaying the transpose (.T) makes it easier to see all columns for the first few rows
print(dollar_df_rich.head())

output_filename = "dollar_df_2025_14_days_w_micro.csv"
# Save the final, powerful dataset to a new CSV file
dollar_df_rich.to_csv(output_filename, index=False)
print(f"\nSuccessfully created and saved the feature-rich DataFrame to '{output_filename}'")