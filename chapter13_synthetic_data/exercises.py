import pandas as pd
# 13.1 Supposeyouareanexecutiontrader.Aclientcallsyouwithanordertocovera
# short position she entered at a price of 100. She gives you two exit conditions:
# profit-taking at 90 and stop-lossat 105.
# (a) Assuming the client believes the price follows an O-U process, are these
# levels reasonable? For what parameters?

# E_0[P] (Long-term mean):
# For the profit target of $90 to be hit with reasonable probability before the stop-loss, E_0[P] should ideally be at or below $90, or at least significantly below the entry price of $100. If E_0[P] was, say, $98, then a $90 profit target might be too ambitious for a mean-reverting process from $100.
# σ (Volatility):
# The distance to the profit target is $100 - $90 = $10.
# The distance to the stop-loss is $105 - $100 = $5.
# If σ is very high (high volatility), the price can swing wildly. A $5 stop-loss might be hit too frequently by random noise, even if the general tendency is mean reversion downwards. A larger σ might necessitate wider stops (and potentially wider profit targets for a similar probability of being hit).
# If σ is very low, the price moves sluggishly. It might take a very long time to reach either $90 or $105. The levels might be too wide if the "jiggles" are tiny.
# φ (Speed of mean reversion):
# If φ is close to 0 (fast mean reversion), the price is strongly pulled towards E_0[P]. If E_0[P] is well below $90, then $90 might be hit relatively quickly.
# If φ is close to 1 (slow mean reversion), the process behaves more like a random walk in the short term. The pull towards the mean is weak, making it harder to predict if $90 will be hit before $105, even if E_0[P] is low. The levels would need to be set considering the typical time horizon for reversion.
# Holding Period: While not explicitly an O-U parameter, the client's implicit desired holding period matters. If the O-U process is slow to revert, these levels might require a long holding period.

# (b) Canyouthinkofanalternative stochastic processunderwhichtheselevels
# make sense?




# 13.2 Fit the time series of dollar bars of E-mini S&P500 futures to an O-U process.
# Given thoseparameters:
dollar_df = pd.read_csv('./data/dollar_df_2025_14_days.csv')
dollar_df['timestamp'] = pd.to_datetime(dollar_df['timestamp'])
dollar_df.set_index('timestamp', inplace=True)



# (a) Produceaheat-mapofSharperatiosforvariousprofit-takingandstop-loss
# levels.

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns # For heatmap
from tqdm import tqdm
import statsmodels.api as sm # For OLS regression
from itertools import product # Used in original batch, though not explicitly here
from random import gauss # Used in original batch

# --- O-U Parameter Estimation Function ---
def estimate_ou_parameters(price_series):
    """
    Estimates Ornstein-Uhlenbeck parameters (phi, sigma, E0_P, hl)
    from a price series using OLS.
    The O-U MOdel is:
    P_t = (1 - φ)E_0[P] + φP_{t-1} + σε_t
    This can be rewritten by subtracting P_{t-1} from both sides:
    P_t - P_{t-1} = (1 - φ)E_0[P] + φP_{t-1} - P_{t-1} + σε_t
    ΔP_t = (1 - φ)E_0[P] - (1 - φ)P_{t-1} + σε_t
    Let θ = (1 - φ). Then φ = 1 - θ.
    The equation becomes: ΔP_t = θE_0[P] - θP_{t-1} + σε_t
    Let α = θE_0[P].
    So, ΔP_t = α - θP_{t-1} + ξ_t (where ξ_t = σε_t is the error term).
    """
    if price_series.isnull().any():
        print("Warning: Price series contains NaNs. Dropping them.")
        price_series = price_series.dropna()
    if len(price_series) < 20: # Need sufficient data
        print("Warning: Price series too short for reliable O-U estimation.")
        # Return some default or NaN values if series is too short
        return np.nan, np.nan, np.nan, np.nan

    delta_p = price_series.diff().dropna()
    p_lagged = price_series.shift(1).dropna()
    
    # Align indices after dropna
    common_index = delta_p.index.intersection(p_lagged.index)
    delta_p_aligned = delta_p.loc[common_index]
    p_lagged_aligned = p_lagged.loc[common_index]

    if len(delta_p_aligned) < 10: # Check after alignment
        print("Warning: Not enough aligned data points for O-U estimation.")
        return np.nan, np.nan, np.nan, np.nan

    X_reg = sm.add_constant(p_lagged_aligned)
    y_reg = delta_p_aligned
    
    try:
        model = sm.OLS(y_reg, X_reg).fit()
        
        alpha_hat = model.params['const']
        theta_hat = -model.params.get(p_lagged_aligned.name, model.params.iloc[1]) # Get by name or position
        
        if theta_hat == 0: # Avoid division by zero
            print("Warning: theta_hat is zero. Cannot estimate E0_P or phi reliably.")
            return np.nan, np.nan, np.nan, np.nan

        phi_hat = 1 - theta_hat
        E0_P_hat = alpha_hat / theta_hat
        
        residuals = model.resid
        sigma_hat = np.std(residuals)
        
        if not (0 < phi_hat < 1): # For stable mean reversion and positive half-life
            # print(f"Warning: Estimated phi_hat ({phi_hat:.4f}) is outside (0,1). Half-life may be invalid.")
            hl_hat = np.nan # Or a very large number if phi_hat is very close to 1
        else:
            hl_hat = -np.log(2) / np.log(phi_hat)
            
    except Exception as e:
        print(f"Error during OLS estimation: {e}")
        return np.nan, np.nan, np.nan, np.nan
        
    return phi_hat, sigma_hat, E0_P_hat, hl_hat

# --- Batch Simulation Function (from your snippet) ---
def batch(coeffs, nIter=1e5, maxHP=100, rPT=np.linspace(.5,10,20),
          rSLm=np.linspace(.5,10,20), seed=0):
    """
    Simulates O-U paths for different PT/SL combinations.
    Assumes a LONG trade: PnL = p_exit - seed.
    Profit target (rPT item) is positive. Stop-loss (rSLm item) is positive magnitude.
    Exits if cP > PT_level or cP < -SL_level.
    """
    if coeffs['hl'] <= 0 or np.isnan(coeffs['hl']): # Invalid half-life
        print(f"Warning: Invalid half-life in coeffs: {coeffs['hl']}. Returning empty results.")
        return []
    if coeffs['sigma'] <= 0 or np.isnan(coeffs['sigma']): # Invalid sigma
        print(f"Warning: Invalid sigma in coeffs: {coeffs['sigma']}. Returning empty results.")
        return []
        
    phi, output1 = 2**(-1./coeffs['hl']), []
    
    for pt_level, sl_level in tqdm(product(rPT, rSLm), total=len(rPT)*len(rSLm), desc="Simulating PT/SL combinations"):
        output2 = []
        for iter_ in range(int(nIter)):
            p, hp = seed, 0 # p starts at the entry price (seed)
            while True:
                # O-U process step: p is the price
                p = (1-phi)*coeffs['forecast'] + phi*p + coeffs['sigma']*gauss(0,1)
                cP = p - seed # Current PnL if it were a long trade entered at 'seed'
                hp += 1
                if cP > pt_level or cP < -sl_level or hp > maxHP: # Conditions for a LONG trade
                    output2.append(cP) # Store PnL of the long trade
                    break
        
        mean_cP = np.mean(output2)
        std_cP = np.std(output2)
        
        if std_cP == 0: # Avoid division by zero
            sharpe = 0.0 if mean_cP == 0 else np.sign(mean_cP) * np.inf
        else:
            sharpe = mean_cP / std_cP
        
        # print(f"PT: {pt_level:.2f}, SL: {sl_level:.2f}, Mean PnL: {mean_cP:.2f}, Std PnL: {std_cP:.2f}, Sharpe: {sharpe:.3f}")
        output1.append((pt_level, sl_level, mean_cP, std_cP, sharpe))
        
    return output1

# --- Main Logic for Exercise 13.2 ---

# 1. Fit the time series of dollar bars to an O-U process
print("\n--- Estimating O-U Parameters ---")
# Use the 'close' price from your dollar bars
price_series_for_ou = dollar_df['close']
phi_estimated, sigma_estimated, E0_P_estimated, hl_estimated = estimate_ou_parameters(price_series_for_ou)

if np.isnan(phi_estimated) or np.isnan(sigma_estimated) or np.isnan(E0_P_estimated) or np.isnan(hl_estimated):
    print("Failed to estimate O-U parameters reliably. Using fallback placeholder values.")
    # Fallback if estimation is problematic (e.g., due to non-stationarity or insufficient data)
    E0_P_estimated = price_series_for_ou.mean()
    sigma_estimated = price_series_for_ou.diff().std()
    if sigma_estimated == 0 or np.isnan(sigma_estimated): sigma_estimated = price_series_for_ou.std() * 0.01 # small fraction if no diff
    hl_estimated = 50 # Assume a moderate half-life (e.g., 50 periods)
    phi_estimated = 2**(-1./hl_estimated)
    print(f"Using Fallback Parameters: E0_P={E0_P_estimated:.2f}, hl={hl_estimated:.2f}, sigma={sigma_estimated:.2f}, phi={phi_estimated:.4f}")
else:
    print(f"Estimated O-U Parameters: E0_P={E0_P_estimated:.2f}, phi={phi_estimated:.4f}, sigma={sigma_estimated:.2f}, hl={hl_estimated:.2f}")


# 'forecast': This represents E_0[P_i,T_i], the long-term mean the process reverts to.
# 'hl': Half-life, which determines phi (speed of mean reversion).
# 'sigma': Volatility of the random shocks.

# Prepare coefficients for the batch simulation
ou_coeffs = {
    'forecast': E0_P_estimated,    # This is the E0[P] for the O-U simulation
    'hl': hl_estimated,            # Estimated half-life
    'sigma': sigma_estimated       # Estimated volatility of shocks
}

# Define the entry point for the simulated trades
# For a mean-reversion strategy, we usually enter when the price deviates from E0_P.
# Let's assume we are testing a SHORT strategy: enter when price is X sigma ABOVE E0_P.
# If you want to test a LONG strategy: enter when price is X sigma BELOW E0_P.
# The interpretation of Sharpe sign will depend on this.
is_short_strategy = False # Set to False if you want to simulate a long entry
entry_sigma_multiple = 1.0  # e.g., enter when 1 sigma away from the mean

if is_short_strategy:
    # Enter short when price is ABOVE the mean, expect it to fall to the mean
    simulation_entry_price = E0_P_estimated + entry_sigma_multiple * sigma_estimated
    print(f"Simulating SHORT trades entered at: {simulation_entry_price:.2f} (expecting reversion to {E0_P_estimated:.2f})")
else:
    # Enter long when price is BELOW the mean, expect it to rise to the mean
    simulation_entry_price = E0_P_estimated - entry_sigma_multiple * sigma_estimated
    print(f"Simulating LONG trades entered at: {simulation_entry_price:.2f} (expecting reversion to {E0_P_estimated:.2f})")


# Define the grid for Profit-Taking (rPT) and Stop-Loss (rSLm) magnitudes
# These are absolute $ amounts of PnL from the entry price
num_grid_pts = 10  # Number of points in PT and SL grids (10x10 = 100 simulations)
max_multiple_sigma_exit = 3.0 # Max PT/SL level in terms of sigma

rPT_levels = np.linspace(0.5 * sigma_estimated, max_multiple_sigma_exit * sigma_estimated, num_grid_pts)
rSLm_levels = np.linspace(0.5 * sigma_estimated, max_multiple_sigma_exit * sigma_estimated, num_grid_pts)

print(f"Profit-Taking levels (abs $): {np.round(rPT_levels,2)}")
print(f"Stop-Loss levels (abs $): {np.round(rSLm_levels,2)}")

# Run the simulations
print("\n--- Running Simulations ---")
simulation_output = batch(coeffs=ou_coeffs,
                          nIter=100000,      # Number of simulated paths per PT/SL combination
                          maxHP=200,       # Maximum holding period for each simulated trade
                          rPT=rPT_levels,
                          rSLm=rSLm_levels,
                          seed=simulation_entry_price) # Entry price for the simulation

if not simulation_output:
    print("Batch simulation returned no output. Exiting.")
else:
    # Reshape results for heatmap
    # Results are (pt_level, sl_level, mean_cP, std_cP, sharpe_raw)
    # sharpe_raw is mean_cP / std_cP (PnL of a long trade)
    
    sharpe_ratios_raw = np.array([res[4] for res in simulation_output])
    
    # Adjust Sharpe sign based on strategy type (long or short)
    if is_short_strategy:
        # For a short strategy, PnL_short = -(p_exit - p_entry) = -cP.
        # Sharpe_short = E[-cP] / Std[-cP] = -E[cP] / Std[cP] = -sharpe_raw.
        sharpe_ratios_strategy = -sharpe_ratios_raw
        print("Adjusted Sharpe ratios for SHORT strategy interpretation.")
    else: # Long strategy
        sharpe_ratios_strategy = sharpe_ratios_raw
        print("Using raw Sharpe ratios for LONG strategy interpretation.")

    try:
        sharpe_matrix = sharpe_ratios_strategy.reshape(len(rPT_levels), len(rSLm_levels))
    except ValueError as e:
        print(f"Error reshaping Sharpe ratios: {e}. Number of results: {len(sharpe_ratios_strategy)}")
        sharpe_matrix = np.full((len(rPT_levels), len(rSLm_levels)), np.nan) # Fallback

    # (a) Produce a heat-map of Sharpe ratios
    print("\n--- (a) Heatmap of Sharpe Ratios ---")
    plt.figure(figsize=(12, 10))
    sns.heatmap(sharpe_matrix, 
                xticklabels=np.round(rSLm_levels,2), 
                yticklabels=np.round(rPT_levels,2), 
                annot=True, fmt=".3f", cmap="viridis_r", cbar_kws={'label': 'Sharpe Ratio'})
    plt.xlabel("Stop-Loss Level (Absolute $ from entry)")
    plt.ylabel("Profit-Taking Level (Absolute $ from entry)")
    strategy_type_str = "SHORT" if is_short_strategy else "LONG"
    plt.title(f"Heatmap of Sharpe Ratios for {strategy_type_str} Mean-Reversion Strategy (O-U Simulation)")
    plt.gca().invert_yaxis() 
    plt.show()

# (b) What is the OTR?
    print("\n--- (b) Optimal Trading Rule (OTR) ---")
    if np.all(np.isnan(sharpe_ratios_strategy)): # Check if all values are NaN
         print("All Sharpe ratios are NaN. Cannot determine OTR.")
    else:
        try:
            max_sharpe_flat_idx = np.nanargmax(sharpe_ratios_strategy) # Ignores NaNs
            # Convert flat index to 2D index
            optimal_pt_idx, optimal_sl_idx = np.unravel_index(max_sharpe_flat_idx, (len(rPT_levels), len(rSLm_levels)))
            
            otr_pt_level = rPT_levels[optimal_pt_idx]
            otr_sl_level = rSLm_levels[optimal_sl_idx]
            otr_sharpe = sharpe_matrix[optimal_pt_idx, optimal_sl_idx] # Get from matrix to ensure it's the adjusted one

            print(f"Optimal Trading Rule (OTR) based on max Sharpe ratio:")
            print(f"  Profit-Taking Level: ${otr_pt_level:.2f}")
            print(f"  Stop-Loss Level    : ${otr_sl_level:.2f}")
            print(f"  Achieved Sharpe Ratio: {otr_sharpe:.3f}")
        except ValueError: # Handles case where sharpe_ratios_strategy might be all NaNs after adjustment
             print("Could not find maximum Sharpe ratio (all values might be NaN).")


