# On a series of bitcoin data form tick, volume and dollar bars.
import datetime
import lakeapi
import pandas as pd
import numpy as np
import cufflinks as cf
import create_bars
import visualize_bars


# Example downlaoded from
# https://colab.research.google.com/drive/1E7MSUT8xqYTMVLiq_rMBLNcZmI_KusK3#scrollTo=hzlJ06LN35lt
# ------------------ Obtain Sample Data -------------------------------------- #
download_data = False

if download_data:
	lakeapi.use_sample_data(anonymous_access = True)

	trades = lakeapi.load_data(
			table="trades",
			start=datetime.datetime(2022, 10, 1),
			end=datetime.datetime(2022, 10, 2),
			symbols=["BTC-USDT"],
			exchanges=['BINANCE'],
	)
	trades.set_index('received_time', inplace = True)

	trades.to_csv('./btc_sample.csv')

# ---------------------------------------------------------------------------- #

# ------------------ Read Data -------------------------------------- #


trades = pd.read_csv('./btc_sample.csv')
trades['received_time'] = pd.to_datetime(trades['received_time'])
trades.set_index('received_time', inplace = True)


cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

print("Creating 15min chart...")

# Regular Plot 15 min chart
# trades['price'].resample('15Min').ohlc().iplot(kind = 'candle', title = '15m candles from trades')
print('Regular Plot ✅')

aggregated_timestamp_bars = create_bars.aggregate_time_bars(trades, time_freq='5min')
print('Regular timestamp aggregate ✅')

# Example usage
aggregated_tick_df = create_bars.aggregate_data(trades, bar_type='tick', threshold=10000)
print('Tick Bars aggregate ✅')

# visualize_bars.plot_aggregated_data(aggregated_tick_df, bar_type='tick')
print('Tick Bars Plot ✅')

aggregated_volume_df = create_bars.aggregate_data(trades, bar_type='volume', threshold=400)
# visualize_bars.plot_aggregated_data(aggregated_volume_df, bar_type='volume')
print('Volume Bars ✅')

aggregated_dollar_df = create_bars.aggregate_data(trades, bar_type='dollar', threshold=8000000)
# visualize_bars.plot_aggregated_data(aggregated_dollar_df, bar_type='dollar')
print('Dollar Bars ✅')


# # Plot tick bars
# Count bars weekly

period_counts = True
if period_counts:
	period_counts_tick = create_bars.count_bars_periodically(aggregated_tick_df, period="15min")
	period_counts_volume = create_bars.count_bars_periodically(aggregated_volume_df, period="15min")
	period_counts_dollar = create_bars.count_bars_periodically(aggregated_dollar_df, period="15min")
	print('Period Counts ✅')

	# Plot bar counts
	visualize_bars.plot_bar_counts(period_counts_tick, bar_type='tick')
	visualize_bars.plot_bar_counts(period_counts_volume, bar_type='volume')
	visualize_bars.plot_bar_counts(period_counts_dollar, bar_type='dollar')


serial_correlation = False
if serial_correlation:
	# Compute serial correlation
	serial_corr_timestamp = create_bars.compute_serial_correlation(aggregated_timestamp_bars)
	serial_corr_tick = create_bars.compute_serial_correlation(aggregated_tick_df)
	serial_corr_volume = create_bars.compute_serial_correlation(aggregated_volume_df)
	serial_corr_dollar = create_bars.compute_serial_correlation(aggregated_dollar_df)

	print(f'Serial Correlation of Timestamp Bars: {serial_corr_timestamp}')
	print(f'Serial Correlation of Tick Bars: {serial_corr_tick}')
	print(f'Serial Correlation of Volume Bars: {serial_corr_volume}')
	print(f'Serial Correlation of Dollar Bars: {serial_corr_dollar}')

	# Determine which bar method has the lowest serial correlation
	bar_types = ['timestamp', 'tick', 'volume', 'dollar']
	serial_correlations = [serial_corr_timestamp, serial_corr_tick, serial_corr_volume, serial_corr_dollar]
	min_serial_corr_index = np.argmin(serial_correlations)

	print(f'The bar method with the lowest serial correlation is {bar_types[min_serial_corr_index]} bars.')


# Partition the bar series into monthly subsets. Compute the variance of returns for every subset of every bar type. Compute the variance of those variances. What method exhibits the smallest variance of variances?
# hourly_timestamp_bars = trades.resample('H').ohlc()
calculate_variances = False
if calculate_variances:
	resample_rules = {
			'open': 'first',
			'high': 'max',
			'low': 'min',
			'close': 'last',
			'volume': 'sum',
	}
	hourly_timestamp_bars = create_bars.aggregate_time_bars(trades, 'H')
	hourly_tick_bars = aggregated_tick_df.resample('H').apply(resample_rules)
	hourly_volume_bars = aggregated_volume_df.resample('H').apply(resample_rules)
	hourly_dollar_bars = aggregated_dollar_df.resample('H').apply(resample_rules)


	#2.27e-06
	hourly_timestamp_bars['returns'] = hourly_timestamp_bars['close'].pct_change()
	variance_of_timestamp_returns = hourly_timestamp_bars['returns'].var()

	#4.42e-06
	hourly_tick_bars['returns'] = hourly_tick_bars['close'].pct_change()
	variance_of_tick_returns = hourly_tick_bars['returns'].var()

	#3.75e-06
	hourly_volume_bars['returns'] = hourly_volume_bars['close'].pct_change()
	variance_of_volume_returns = hourly_volume_bars['returns'].var()

	#3.35e-06
	hourly_dollar_bars['returns'] = hourly_dollar_bars['close'].pct_change()
	variance_of_dollar_returns = hourly_dollar_bars['returns'].var()

	# Variance of Variances: 8.110916666666669e-13

	import scipy.stats as stats

	stats.jarque_bera(list(hourly_dollar_bars['returns']))

	









