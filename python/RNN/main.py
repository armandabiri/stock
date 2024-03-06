{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, LSTM, Dropout\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "# Function to prepare data for LSTM\n",
    "def prepare_data(data, features, time_steps):\n",
    "    \"\"\"\n",
    "    Prepares the data for LSTM by creating input sequences and corresponding output values.\n",
    "\n",
    "    Parameters:\n",
    "    - data: Input data array containing features\n",
    "    - features: List of feature names\n",
    "    - time_steps: Number of time steps to consider for each sequence\n",
    "\n",
    "    Returns:\n",
    "    - X: Input sequences\n",
    "    - y: Corresponding output values\n",
    "    \"\"\"\n",
    "    X, y = [], []\n",
    "    for i in range(len(data) - time_steps - 1):\n",
    "        X.append(data[i:(i + time_steps), :])\n",
    "        y.append(data[i + time_steps, 0])  # Considering only the 'Close' price as output\n",
    "    return np.array(X), np.array(y)\n",
    "\n",
    "# Load your stock data\n",
    "# Assuming you have a DataFrame df with features: 'Close', 'RSI', 'Median', 'MACD', 'Volume', etc.\n",
    "# You may need to calculate these features from your raw data if not available\n",
    "# Example: df['RSI'] = calculate_rsi(df['Close'], window=14)\n",
    "# Example: df['MACD'] = calculate_macd(df['Close'], 12, 26, 9)\n",
    "\n",
    "# Select features and normalize data\n",
    "features = ['Close', 'RSI', 'Median', 'MACD', 'Volume']\n",
    "scaler = MinMaxScaler(feature_range=(0, 1))\n",
    "data = scaler.fit_transform(df[features])\n",
    "\n",
    "# Set time steps for LSTM\n",
    "time_steps = 10\n",
    "\n",
    "# Prepare data\n",
    "X, y = prepare_data(data, features, time_steps)\n",
    "\n",
    "# Split data into training and testing sets\n",
    "split = int(0.8 * len(data))\n",
    "X_train, X_test = X[:split], X[split:]\n",
    "y_train, y_test = y[:split], y[split:]\n",
    "\n",
    "# Build LSTM model\n",
    "model = Sequential([\n",
    "    LSTM(units=50, return_sequences=True, input_shape=(time_steps, len(features))),\n",
    "    Dropout(0.2),  # Dropout layer for regularization\n",
    "    LSTM(units=50, return_sequences=True),\n",
    "    Dropout(0.2),\n",
    "    LSTM(units=50),\n",
    "    Dropout(0.2),\n",
    "    Dense(units=1)  # Output layer\n",
    "])\n",
    "\n",
    "# Compile model\n",
    "model.compile(optimizer='adam', loss='mean_squared_error')\n",
    "\n",
    "# Train model\n",
    "model.fit(X_train, y_train, epochs=100, batch_size=64)\n",
    "\n",
    "# Make predictions\n",
    "train_predictions = model.predict(X_train)\n",
    "test_predictions = model.predict(X_test)\n",
    "\n",
    "# Inverse transform predictions\n",
    "train_predictions = scaler.inverse_transform(np.hstack((train_predictions, X_train[:, -1, 1:])))\n",
    "y_train_inv = scaler.inverse_transform([[x] for x in y_train])\n",
    "test_predictions = scaler.inverse_transform(np.hstack((test_predictions, X_test[:, -1, 1:])))\n",
    "y_test_inv = scaler.inverse_transform([[x] for x in y_test])\n",
    "\n",
    "# Calculate RMSE\n",
    "train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predictions[:, 0]))\n",
    "test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predictions[:, 0]))\n",
    "print(f'Train RMSE: {train_rmse}')\n",
    "print(f'Test RMSE: {test_rmse}')\n",
    "\n",
    "# Visualize predictions\n",
    "plt.plot(df.index[:len(train_predictions)], y_train_inv, label='Training Data')\n",
    "plt.plot(df.index[len(train_predictions)+time_steps+1:], y_test_inv, label='Testing Data')\n",
    "plt.plot(df.index[:len(train_predictions)], train_predictions[:, 0], label='Train Predictions')\n",
    "plt.plot(df.index[len(train_predictions)+time_steps+1:], test_predictions[:, 0], label='Test Predictions')\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
