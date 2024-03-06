import pandas as pd
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from collections import Counter


class StockClassifier:
    def __init__(self, csvfile='data/NASDAQ-100_Close.csv'):
        self.csvfile = csvfile

    def process_data_for_labels(self, ticker, history_days):
        df = pd.read_csv(self.csvfile, index_col=0)
        tickers = df.columns.tolist()
        df.fillna(0, inplace=True)
        for i in range(1, history_days + 1):
            df[f'{ticker}_{i}d'] = (df[ticker].shift(-i) - df[ticker]) / df[ticker] * 100
        df.fillna(0, inplace=True)
        return tickers, df

    def buy_sell_hold(self, *args):
        requirement = 10
        for col in args:
            if any(col > requirement):
                return 1
            if any(col < -requirement):
                return -1
        return 0

    def extract_feature_sets(self, ticker, history_days):
        tickers, df = self.process_data_for_labels(ticker, history_days)
        df[f'{ticker}_target'] = df[[f'{ticker}_{i}d' for i in range(1, history_days+1)]].apply(self.buy_sell_hold, axis=1)
        df.dropna(inplace=True)

        df_vals = df[tickers].pct_change()
        df_vals.fillna(0, inplace=True)

        X = df_vals.values
        y = df[f'{ticker}_target'].values

        return X, y, df

    def do_ml(self, ticker, history_days=7):
        X, y, _ = self.extract_feature_sets(ticker, history_days)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

        clf = VotingClassifier([('lsvc', svm.LinearSVC()), ('knn', neighbors.KNeighborsClassifier()),
                                ('rfor', RandomForestClassifier())])
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        predictions = clf.predict(X_test)

        return accuracy, Counter(predictions)

def main():
    ticker='AAPL'
    stock_classifier = StockClassifier()
    accuracy, predicted_spread = stock_classifier.do_ml(ticker)
    print('Accuracy:', accuracy)
    print('Predicted spread:', predicted_spread)

if __name__ == '__main__':
    main()


