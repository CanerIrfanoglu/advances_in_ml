import pandas as pd

def aggregate_data(df, bar_type='tick', threshold=10000):
		# Initialize lists to hold the aggregated data
		timestamps = []
		open_prices = []
		high_prices = []
		low_prices = []
		close_prices = []
		vwap_values = []
		volumes = []

		if bar_type == 'tick':
				num_bars = len(df) // threshold
				for i in range(num_bars):
						tick_data = df.iloc[i * threshold : (i + 1) * threshold]
						open_price = tick_data['price'].iloc[0]
						high_price = tick_data['price'].max()
						low_price = tick_data['price'].min()
						close_price = tick_data['price'].iloc[-1]
						vwap_value = (tick_data['price'] * tick_data['quantity']).sum() / tick_data['quantity'].sum()
						total_volume = tick_data['quantity'].sum()
						timestamps.append(tick_data.index[0])
						open_prices.append(open_price)
						high_prices.append(high_price)
						low_prices.append(low_price)
						close_prices.append(close_price)
						vwap_values.append(vwap_value)
						volumes.append(total_volume)
		
		elif bar_type == 'volume':
				current_volume = 0
				start_index = 0
				for i in range(len(df)):
						current_volume += df['quantity'].iloc[i]
						if current_volume >= threshold:
								tick_data = df.iloc[start_index : i + 1]
								open_price = tick_data['price'].iloc[0]
								high_price = tick_data['price'].max()
								low_price = tick_data['price'].min()
								close_price = tick_data['price'].iloc[-1]
								vwap_value = (tick_data['price'] * tick_data['quantity']).sum() / tick_data['quantity'].sum()
								total_volume = tick_data['quantity'].sum()
								timestamps.append(tick_data.index[0])
								open_prices.append(open_price)
								high_prices.append(high_price)
								low_prices.append(low_price)
								close_prices.append(close_price)
								vwap_values.append(vwap_value)
								volumes.append(total_volume)
								current_volume = 0
								start_index = i + 1
		
		elif bar_type == 'dollar':
				current_dollar = 0
				start_index = 0
				for i in range(len(df)):
						current_dollar += df['price'].iloc[i] * df['quantity'].iloc[i]
						if current_dollar >= threshold:
								tick_data = df.iloc[start_index : i + 1]
								open_price = tick_data['price'].iloc[0]
								high_price = tick_data['price'].max()
								low_price = tick_data['price'].min()
								close_price = tick_data['price'].iloc[-1]
								vwap_value = (tick_data['price'] * tick_data['quantity']).sum() / tick_data['quantity'].sum()
								total_volume = tick_data['quantity'].sum()
								timestamps.append(tick_data.index[0])
								open_prices.append(open_price)
								high_prices.append(high_price)
								low_prices.append(low_price)
								close_prices.append(close_price)
								vwap_values.append(vwap_value)
								volumes.append(total_volume)
								current_dollar = 0
								start_index = i + 1

		aggregated_df = pd.DataFrame({
				'timestamp': timestamps,
				'open': open_prices,
				'high': high_prices,
				'low': low_prices,
				'close': close_prices,
				'VWAP': vwap_values,
				'volume': volumes
		})

		return aggregated_df

def aggregate_time_bars(df, time_freq='5Min'):
		
		if not df.index.name == "timestamp":
			df['timestamp'] = pd.to_datetime(df.index)
			# Set timestamp as the index
			df.set_index('timestamp', inplace=True)

		ohlc_dict = {
				'price': 'ohlc',
				'quantity': 'sum'
		}
		aggregated_df = df.resample(time_freq).apply(ohlc_dict)
		# Flatten the MultiIndex columns
		aggregated_df.columns = ['open', 'high', 'low', 'close', 'volume']
		aggregated_df.dropna(inplace=True)
		return aggregated_df


def count_bars_periodically(aggregated_df, period = "weekly"):
		if not aggregated_df.index.name == "timestamp":
			aggregated_df['timestamp'] = pd.to_datetime(aggregated_df['timestamp'])
			# Set timestamp as the index
			aggregated_df.set_index('timestamp', inplace=True)
		
		# Count the number of bars per week
		if period == "weekly":
				resample_period = "W"
		elif period == "monthly":
				resample_period = "M"
		elif period == "hourly":
				resample_period = "H"
		else:
				resample_period = period

		bar_counts = aggregated_df.resample(resample_period).size()
		print(f"With {period} period, there are total {sum(bar_counts)} bars between {min(aggregated_df.index)} and {max(aggregated_df.index)}")
		return bar_counts

def calculate_returns(df):
		df['returns'] = df['close'].pct_change()
		return df

def compute_serial_correlation(df):
		df = calculate_returns(df)
		df = df.dropna(subset=['returns'])
		return df['returns'].autocorr(lag=1)


