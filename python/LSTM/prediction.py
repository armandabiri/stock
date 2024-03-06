import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Input
import matplotlib.dates as mdates
import mplcursors

class StockPredictor:
    def __init__(self, csvfile='data/NASDAQ-100_Close.csv', model='LSTM', ticker=None):
        self.stock_model = os.path.splitext(os.path.basename(csvfile))[0]
        self.model = model
        self.df = None
        self.filename = f'models/{self.model}_{self.stock_model}.keras'
        self.df = pd.read_csv(csvfile)
        self.values_scaled = None
        self.model = None
        self.time_frame = None
        self.time_horizon = None
        self.ticker=ticker
        self.tickers=self.df.columns[1:]
        self.Scaler = MinMaxScaler(feature_range=(0, 1))


    def prep_data(self, time_frame=10, time_horizon=10, split_proportion=0.8):
        # Convert timestamp column to datetime format
        self.df['timestamp'] = pd.to_datetime(self.df['Date'])
        # Calculate the step size (time difference) between consecutive timestamps
        self.df['step_size'] = self.df['timestamp'].diff().dt.total_seconds()
        # Fill any NaN values in step_size column
        self.df['step_size'].fillna(0, inplace=True)  # For the first data point, step size is assumed to be 0

        self.time_frame = time_frame
        self.time_horizon = time_horizon

        size=self.df.shape[0]
        if self.tickers[0]=='Open':
            close_values = self.df['Close'].values.reshape(-1, 1)
            self.features = close_values
            # volume_values = self.df['Volume'].values.reshape(-1, 1)
            # self.features = np.concatenate((close_values, volume_values), axis=1)
        else:
            self.features = self.df[self.tickers].values


        self.scales = tuple(np.max(self.features, axis=0))
        normalized_features = self.features / self.scales
        scaled_features = self.Scaler.fit_transform(normalized_features)

        self.inputs=[]
        for i in range(time_frame, size-1-time_horizon):
            self.inputs.append(scaled_features[i - time_frame:i, :])

        self.outputs=[]
        for i in range(time_frame, size-1-time_horizon):
            self.outputs.append(scaled_features[1+i:i+time_horizon+1, :])


        self.inputs = np.array(self.inputs)
        self.outputs = np.array(self.outputs)
        self.outputs=self.outputs.reshape(self.outputs.shape[0], self.outputs.shape[1]*self.outputs.shape[2])

        split_index = int( size* split_proportion)

        self.x_train=self.inputs[:split_index,:,:]
        self.y_train=self.outputs[:split_index,:]
        self.x_test=self.inputs[split_index:,:,:]
        self.y_test=self.outputs[split_index:,:]


    def train(self, neurons=[48, 48, 48], dropout=[0.2], epochs=3, batch_size=32):
        if len(neurons) != len(dropout):
            dropout = [dropout[0]] * len(neurons)

        model = Sequential()
        model.add(Input(shape=(self.x_train.shape[1], self.x_train.shape[2])))

        for i in range(len(neurons) - 1):
            model.add(LSTM(units=neurons[i], return_sequences=True))
            model.add(Dropout(dropout[i]))

        model.add(LSTM(units=neurons[-1]))
        model.add(Dropout(dropout[-1]))

        model.add(Dense(units=self.y_train.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)
        model.save(self.filename, overwrite=True)
        self.model = model


    def predict(self, days=200):
        if not self.model:
            model = keras.models.load_model(self.filename)
        else:
            model = self.model

        y_test = model.predict(self.x_test)
        print(model.evaluate(self.x_test, self.y_test))
        y_test = self.Scaler.inverse_transform(y_test)
        y_test = y_test*self.scales

        X = self.x_test[-1, :,:].reshape(1, self.x_test.shape[1], self.x_test.shape[2])
        y_pred = []
        for i in range(0, days):
            y = model.predict(X)
            if X.shape[2]==1:
                X = np.append(X, y[0,0].reshape(1, 1, 1), axis=1)
                y_pred.append(self.Scaler.inverse_transform(y)*self.scales)
            else:
                X = np.append(X, y.reshape(1, 1, -1), axis=1)
                y_pred.append(self.Scaler.inverse_transform(y)*self.scales)
            X = X[:, 1:, :]



        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        y_pred=y_pred.reshape(-1, y_pred.shape[2])


        dates = pd.to_datetime(self.df['Date'])
        date_pred = pd.date_range(dates.iloc[-self.time_horizon], periods=days)
        date_test = dates[-len(self.y_test):]

        fig, ax = plt.subplots(figsize=(8, 4))
        plt.plot(dates, self.features[:,0], color='black', label="True Price")
        plt.plot(date_test, y_test[:, 0], color='blue', label='Predicted Testing Price')
        plt.plot(date_pred, y_pred[:, 0], color='red', label='Predicted Future Price')

        plt.legend()

        # Set x-axis ticks and labels for each month
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Set x-axis ticks and labels for each year
        plt.xticks(rotation=90)
        plt.xlabel('Date')
        plt.ylabel('Price')

        # Enable mplcursors for the plot
        mplcursors.cursor(hover=True)

        # Add annotations or tooltips
        # @mplcursors.cursor(ax, hover=True)
        def on_hover(sel):
            x, y = sel.target
            sel.annotation.set_text(f'x={x}, y={y}')

        plt.show()


def main():
    stock_predictor = StockPredictor(csvfile='data/ABNB.csv')
    stock_predictor.prep_data(time_frame=56, time_horizon=14, split_proportion=0.8)
    # stock_predictor.train(neurons=[96, 96, 96], dropout=[0.25], epochs=20, batch_size=32)
    stock_predictor.predict(days=356)


if __name__ == '__main__':
    main()
