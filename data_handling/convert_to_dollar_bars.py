import pandas as pd
from datetime import datetime, timezone # For timestamp conversion if not using pandas' vectorized way directly

btc_data_raw = pd.read_csv('./data/BTCUSDT_combined_may2025_1_to_14.csv')

def convert_ms_epoch_to_strftime(ms_epoch_series):
    """
    Converts a pandas Series of millisecond epoch timestamps to
    'YYYY-MM-DD HH:MM:SS.ffffff' string format, ensuring UTC.
    """
    # Convert milliseconds to seconds, then to datetime objects (UTC)
    dt_series = pd.to_datetime(ms_epoch_series / 1000, unit='s', utc=True)
    # Format to the desired string
    return dt_series.dt.strftime('%Y-%m-%d %H:%M:%S.%f')

transformed_df = pd.DataFrame()

transformed_df['received_time'] = convert_ms_epoch_to_strftime(btc_data_raw['timestamp'])
transformed_df['side'] = btc_data_raw["side"]
transformed_df['quantity'] = btc_data_raw["volume"]
transformed_df['price'] = btc_data_raw["price"]



# Convert to dollar bars
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
						timestamps.append(tick_data["received_time"].iloc[0])
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
								timestamps.append(tick_data["received_time"].iloc[0])
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
								timestamps.append(tick_data["received_time"].iloc[0])
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

dollar_df = aggregate_data(transformed_df, "dollar", 1000000)