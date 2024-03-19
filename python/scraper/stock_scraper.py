import pickle
import requests
import os
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pandas_datareader import data as pdr
import yfinance as yf
from pathlib import Path
import json

style.use('ggplot')



def get_pickle_files(folder_path):
    pickle_files = []

    # Iterate over all files in the folder
    for file in os.listdir(folder_path):
        # Check if the file ends with ".pickle"
        if file.endswith(".pickle"):
            pickle_files.append(os.path.join(folder_path, file))
    return pickle_files

def download_tickers(jsonfile="stocks.json", directory="data"):
    def read_url(url, symbols):
        tickers = symbols.copy()
        if url != "":
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code != 200:
                print("Failed to fetch data from the webpage.")
                return []

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", {"class": "wikitable"})

            if not table:
                print("Unable to find the table in the webpage.")
                return []

            # Extract tickers from the table
            for row in table.find_all("tr")[1:]:
                ticker = row.find_all("td")[0].text.strip()
                tickers.append(ticker)
        tickers = list(set(tickers))  # Convert to set and then back to list to ensure uniqueness
        return tickers

    # Read JSON file
    if not os.path.exists(jsonfile):
        print(f"File {jsonfile} not found.")
        return

    with open(jsonfile, "r") as f:
        data = json.load(f)

    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    for index, index_data in data.items():
        symbols = index_data.get("symbols", [])
        url = index_data.get("url", "")
        tickers = read_url(url, symbols)

        # Save the tickers to a file using pickle
        picklefile = f'{directory}/{index}.pickle'
        with open(picklefile, "wb") as f:
            pickle.dump(tickers, f)

        print(f"Tickers for {index} saved successfully.")


def download_stock(picklefile="data/SP500.pickle", start_date="2024-01-30", end_date="2024-04-30", update_data=False):
    # Ensure the directory exists or create it
    directory = os.path.dirname(picklefile)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Load tickers from the pickle file
    with open(picklefile, "rb") as f:
        tickers = pickle.load(f)

    # Loop through each ticker
    for count, ticker in enumerate(tickers, start=1):
        csv_path = f'{directory}/{ticker}.csv'
        # Check if data file exists for the ticker
        try:
            yf.pdr_override() # <== that's all it takes :-)
            data = pdr.get_data_yahoo(ticker, start_date, end_date)
            data.to_csv(csv_path)
            print(f"Downloaded data for {ticker} and saved as {ticker}.csv - {count}/{len(tickers)}")
        except Exception as e:
            print(f"Failed to retrieve data for {ticker}: {e} - {count}/{len(tickers)}")



def combine_data(picklefile):
    directory = os.path.dirname(picklefile)
    file_name = os.path.splitext(os.path.basename(picklefile))[0]
    with open(picklefile, "rb") as f:
        tickers = pickle.load(f)

    # Initialize an empty DataFrame to store compiled data
    for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
        main_df = pd.DataFrame()
        for count, ticker in enumerate(tickers, start=1):
            # Read data from CSV file
            csv_path = Path(f'{directory}/{ticker}.csv')
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
                df.rename(columns={'Adj Close': ticker}, inplace=True)
                main_df[ticker] = df[feature]  # Add column directly without joining
                print(f"Compiling: {count}/{len(tickers)}")
            else:
                print(f"File not found for ticker {ticker}. Skipping.")
        # Save the compiled DataFrame to CSV file
        main_df.to_csv(f'{directory}/{file_name}_{feature}.csv')


def visualize_data(csvfile):
    df = pd.read_csv(csvfile, index_col=0)

    # Normalize the data from 0 to 1
    normalized_df = (df - df.min()) / (df.max() - df.min())

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot normalized stock prices in the first subplot
    ax1 = axes[0, 0]
    for column in normalized_df.columns:
        ax1.plot(normalized_df.index, normalized_df[column], label=column)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price')
    ax1.set_title('Normalized Stock Prices')
    ax1.legend()

    # Plot correlation in the second subplot
    ax2 = axes[0, 1]
    corr_matrix = df.corr()
    heatmap = ax2.imshow(corr_matrix, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap, ax=ax2)
    ax2.set_xticks(np.arange(len(df.columns)))
    ax2.set_yticks(np.arange(len(df.columns)))
    ax2.set_xticklabels(df.columns, rotation=90)
    ax2.set_yticklabels(df.columns)
    ax2.set_title('Correlation Matrix')

    # Plot cross-correlation in the third subplot
    ax3 = axes[1, 0]
    cross_corr_matrix = df.corr(method='pearson').abs()
    cross_corr_matrix = cross_corr_matrix.fillna(0)  # Fill NaN values with 0
    heatmap2 = ax3.imshow(cross_corr_matrix, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap2, ax=ax3)
    ax3.set_xticks(np.arange(len(df.columns)))
    ax3.set_yticks(np.arange(len(df.columns)))
    ax3.set_xticklabels(df.columns, rotation=90)
    ax3.set_yticklabels(df.columns)
    ax3.set_title('Cross-Correlation Matrix')

    # Remove the fourth subplot
    fig.delaxes(axes[1, 1])

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def main():
    download_tickers(jsonfile="python/scraper/stocks.json",directory="data")

    folder="data"
    start_date = "2018-01-01"
    end_date = "2024-04-06"

    pickle_files=get_pickle_files(folder)
    for picklefile in pickle_files:
        download_stock(f'{picklefile}',start_date,end_date,update_data=True)
        combine_data(picklefile)
    visualize_data('data/NASDAQ-100_Close.csv')

if __name__ == "__main__":
    main()