import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from yahoo_fin import stock_info as si
import mplcursors

class StockStatistics:
    def __init__(self, data):
        self.data = data

    def rsi(self, window=14, open_or_close='close'):
        # Calculate price differences
        delta = self.data[open_or_close].diff()
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        # Calculate average gains and losses using convolution
        kernel = np.ones(window) / window
        avg_gain = np.convolve(gain.values, kernel, mode='valid')
        avg_loss = np.convolve(loss.values, kernel, mode='valid')
        # Calculate relative strength (RS)
        rs = avg_gain / avg_loss
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def macd(self, short_window=12, long_window=26, signal_window=9, open_or_close='close'):
        # Calculate short EMA
        short_ema = self.data[open_or_close].ewm(span=short_window, min_periods=1, adjust=False).mean()
        # Calculate long EMA with the same starting point as short EMA
        long_ema = self.data[open_or_close].iloc[short_window-long_window:].ewm(span=long_window, min_periods=1, adjust=False).mean()
        # Calculate MACD line
        macd_line = short_ema - long_ema
        # Calculate signal line using convolution
        signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()
        return macd_line, signal_line

    def ema(self, data, window):
        # Exponential moving average
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        ema = np.convolve(data, weights, mode='valid')
        return ema

    def mean(self, window=5, open_or_close='close'):
        kernel = np.ones(window) / window
        mean_signal = np.convolve(self.data[open_or_close].values, kernel, mode='valid')
        return mean_signal

    def std(self, window=5, open_or_close='close'):
        # Calculate the mean signal using convolution
        mean_signal = self.mean(window, open_or_close)
        # Calculate the squared differences
        squared_diff = (self.data[open_or_close].values[window-1:] - mean_signal)**2
        # Convolve the squared differences with a kernel of ones divided by window size
        kernel = np.ones(window) / window
        mean_squared_diff = np.convolve(squared_diff, kernel, mode='valid')
        # Take square root of the convolved signal to get the standard deviation signal
        std_signal = np.sqrt(mean_squared_diff)
        return std_signal

    def median(self, window=5, open_or_close='close'):
        median_signal = np.zeros(len(self.data) - window + 1)
        for i in range(len(self.data) - window + 1):
            window_data = self.data[open_or_close].values[i:i+window]
            median_signal[i] = np.median(window_data)
        return median_signal



def main():
    # Define the ticker symbol
    ticker = "SMCI"  # Example: Apple Inc.

    # Retrieve historical data
    df = si.get_data(ticker)

    stats = StockStatistics(df)
    rsi = stats.rsi()
    macd_line, macd_signal = stats.macd()
    mean_close = stats.mean(window=5)

    times=df.index
    N=len(times)


    # Plot stock data with indicators
    plt.figure(figsize=(12, 8))
    plt.plot(times, df['close'], label='close Price')
    plt.plot(times[N-len(mean_close):], mean_close, label='5-Day Mean close Price', linestyle='--')
    # plt.plot(times[N-len(rsi):], rsi, label='RSI')
    # plt.plot(times[N-len(macd_line):], macd_line, label='MACD Line')
    # plt.plot(times[N-len(macd_signal):], macd_signal, label='MACD Signal Line')


    # plt.xlabel('Date')
    # plt.ylabel('Price/Value')
    # plt.title('Stock Data with Indicators')
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(rotation=45)
    # plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # plt.tight_layout()

    mplcursors.cursor(hover=True) # Activate mouse scroll zooming
    plt.show()


if __name__ == "__main__":
    main()
