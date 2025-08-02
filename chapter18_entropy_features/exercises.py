# Chat LINK: https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221UuXpFx7G4eI4dTzFPYZlnPOibgguzzrv%22%5D,%22action%22:%22open%22,%22userId%22:%22111971824642554314977%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# --- Data Loading and Initial Setup (from your script) ---
try:
    df = pd.read_csv('./data/dollar_df_2025_14_days.csv', index_col=0)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
except FileNotFoundError:
    print("File not found. Using a longer dummy dataframe to satisfy exercise 18.2 requirements.")
    # For exercise 18.2, we need a very long series (1000 subsets of 100 = 100,000 points)
    dates = pd.to_datetime(pd.date_range(start='2025-01-01', periods=120000, freq='T'))
    prices = 100 + np.random.randn(120000).cumsum() * 0.05
    df = pd.DataFrame(data={'close': prices}, index=dates)

print(f"Original 'close' column contains {df['close'].isnull().sum()} NaN values.")
close_prices = df['close'].dropna()
print(f"Working with {len(close_prices)} non-NaN price points.")

# Create a clean returns series for all exercises
log_price = np.log(close_prices)
# IMPORTANT: We will use this 'returns' series for ALL exercises. It has one NaN at the start.
returns = log_price.diff()


# ==============================================================================
# --- SOLUTIONS FOR EXERCISE 18.1 ---
# ==============================================================================
print("\n" + "="*50)
print("--- Starting Exercise 18.1 ---")
print("="*50)

# We use the 'returns' series and drop NaNs for each specific encoding.

# --- 18.1 (a) Binary Quantization ---
print("\n(a) Binary Quantization")
binary_returns = (returns > 0).astype(int).dropna()
print("Value counts:\n", binary_returns.value_counts())

# --- 18.1 (b) Quantile Encoding (10 Letters) ---
print("\n(b) Quantile Encoding with 10 Bins")
quantile_encoded_returns = pd.qcut(returns.dropna(), q=10, labels=False, duplicates='drop')
print("Value counts (should be roughly equal):\n", pd.Series(quantile_encoded_returns).value_counts().sort_index())

# --- 18.1 (c) Sigma Encoding ---
print("\n(c) Sigma Encoding")
sigma = returns.std()
bin_edges = [-np.inf, -2*sigma, -sigma, 0, sigma, 2*sigma, np.inf]
sigma_encoded_returns = pd.cut(returns.dropna(), bins=bin_edges, labels=False, right=False)
print("Value counts (should be concentrated in the middle):\n", pd.Series(sigma_encoded_returns).value_counts().sort_index())

# --- 18.1 (d) & (e) Helper Functions ---

def plugin_entropy(series):
    """Calculates Shannon entropy using the plug-in estimator."""
    counts = pd.Series(series).value_counts()
    if len(counts) <= 1: return 0 # No uncertainty if only one symbol
    probs = counts / len(series)
    return -np.sum(probs * np.log2(probs))

def lz77_complexity(string_representation):
    """Calculates the number of phrases in the LZ77 parsing of a string."""
    i, phrases = 0, set()
    while i < len(string_representation):
        longest_match = ""
        for j in range(i, len(string_representation)):
            substring = string_representation[i:j+1]
            if substring in phrases:
                longest_match = substring
            else:
                phrases.add(substring)
                break
        i += len(longest_match) + 1
    return len(phrases)

def kontoyiannis_entropy_single(series_data):
    """Calculates the Kontoyiannis entropy estimate for a single sequence."""
    n = len(series_data)
    if n < 2: return 0
    string_representation = ''.join(map(str, series_data))
    lz_phrases = lz77_complexity(string_representation)
    return (lz_phrases / n) * np.log2(n)

# ==============================================================================
# --- SOLUTIONS FOR EXERCISE 18.2 ---
# ==============================================================================

print("\n" + "="*50)
print("--- Starting Exercise 18.2 ---")
print("="*50)

# (a) Compute the returns series, {r_t}
# We will use the 'returns' variable calculated at the beginning.
print(f"(a) Using the returns series of length {len(returns)} calculated earlier.")

# (b) Encode the series based on momentum/reversal
print("\n(b) Encoding the series for Momentum (1) vs. Reversal (0)...")
# r_t * r_{t-1} >= 0 means returns have the same sign (momentum) -> 1
# r_t * r_{t-1} < 0 means returns have different signs (reversal) -> 0
product_of_returns = returns * returns.shift(1)
encoded_series = (product_of_returns >= 0).astype(int)
encoded_series = encoded_series.dropna() # Drop the NaN from the shift() operation

print("First 10 encoded values:", encoded_series.head(10).tolist())
print("Value counts for the entire series:\n", encoded_series.value_counts())

# (c) Partition the series into 100 non-overlapping subsets
print("\n(c) Partitioning the series...")
num_subsets = 100
subset_size = 100 # From part (e), the window size is 100
total_len_needed = num_subsets * subset_size

if len(encoded_series) < total_len_needed:
    print(f"\n--- WARNING ---")
    print(f"Data is too short for this exercise.")
    print(f"Need at least {total_len_needed} data points, but only have {len(encoded_series)}.")
    print(f"The following steps will not be executed with your current data file.")
    print(f"To run the full script, a longer data series (or the dummy data) is required.")
else:
    # Drop observations from the beginning to get an exact multiple
    start_index = len(encoded_series) - total_len_needed
    trimmed_series = encoded_series.iloc[start_index:]
    
    # Reshape the 1D series into a 2D array of (num_subsets x subset_size)
    subsets = trimmed_series.values.reshape((num_subsets, subset_size))
    print(f"Successfully created {subsets.shape[0]} subsets of size {subsets.shape[1]}.")

    # (d) Compute the entropy of each subset using the plug-in method
    print("\n(d) Computing plug-in entropy for each subset...")
    plugin_entropies = [plugin_entropy(subset) for subset in subsets]
    print(f"Computed {len(plugin_entropies)} plug-in entropy values.")
    print(f"First 5 plug-in entropy values: {[f'{e:.4f}' for e in plugin_entropies[:5]]}")

    # (e) Compute the entropy of each subset using the Kontoyiannis method
    print("\n(e) Computing Kontoyiannis (LZ) entropy for each subset...")
    lz_entropies = [kontoyiannis_entropy_single(subset) for subset in subsets]
    print(f"Computed {len(lz_entropies)} LZ entropy values.")
    print(f"First 5 LZ entropy values: {[f'{e:.4f}' for e in lz_entropies[:5]]}")

    # (f) Compute the correlation between results 2.d and 2.e
    print("\n(f) Computing the correlation between the two entropy series...")
    
    # Create pandas Series for easy correlation calculation and plotting
    plugin_series = pd.Series(plugin_entropies, name='Plugin_Entropy')
    lz_series = pd.Series(lz_entropies, name='LZ_Entropy')
    
    correlation = plugin_series.corr(lz_series)
    
    print(f"\n>>>> The correlation between plug-in and LZ entropy is: {correlation:.4f} <<<<")

    # Bonus: Visualize the correlation
    plt.figure(figsize=(10, 7))
    plt.scatter(plugin_series, lz_series, alpha=0.5, edgecolors='k', linewidth=0.5)
    plt.title('Correlation between Plug-in and LZ Entropy Estimators\n(for Momentum/Reversal Series)', fontsize=16)
    plt.xlabel('Plug-in Entropy (Measures Randomness of Frequency)', fontsize=12)
    plt.ylabel('Kontoyiannis (LZ) Entropy (Measures Randomness of Sequence)', fontsize=12)
    plt.grid(linestyle='--', alpha=0.6)
    plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    plt.show()


# Rest of the exercises are not solved as 18.3 & 18.4 was repetitive in the sense of
# applying the same entropy estimators to syntheticly generated normal data.
# And 18.5 is the application of entropy in a portfolio management context which is
# beyond the scope of using entropy for trading purposes.