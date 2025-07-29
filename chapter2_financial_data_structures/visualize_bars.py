import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_aggregated_data(aggregated_df, bar_type='tick'):
    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.1, 
        row_heights=[0.7, 0.3]
    )

    # Add Candlestick chart
    fig.add_trace(go.Candlestick(
        x=aggregated_df['timestamp'],
        open=aggregated_df['open'],
        high=aggregated_df['high'],
        low=aggregated_df['low'],
        close=aggregated_df['close'],
        name='Price'
    ), row=1, col=1)

    # Add VWAP line chart
    fig.add_trace(go.Scatter(
        x=aggregated_df['timestamp'],
        y=aggregated_df['VWAP'],
        mode='lines',
        name='VWAP',
        line=dict(color='blue'),
        opacity=0.6
    ), row=1, col=1)

    # Add Volume bar chart
    fig.add_trace(go.Bar(
        x=aggregated_df['timestamp'],
        y=aggregated_df['volume'],
        name='Volume',
        marker_color='rgba(158,202,225,0.5)',
        marker_line_color='rgba(8,48,107,0.8)',
        marker_line_width=1.5,
        opacity=0.3
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'Aggregated {bar_type.capitalize()} Data with Volume',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis2_title='Time',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False
    )

    fig.show()
    


def plot_bar_counts(bar_counts, bar_type='tick'):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=bar_counts.index,
        y=bar_counts.values,
        mode='lines+markers',
        name=f'{bar_type.capitalize()} Bar Counts'
    ))
    
    fig.update_layout(
        title=f'{bar_type.capitalize()} Bar Counts',
        xaxis_title='Time',
        yaxis_title='Number of Bars'
    )
    
    fig.show()