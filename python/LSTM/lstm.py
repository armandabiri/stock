import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
from plot_model import plot_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def plots(*args, **kwargs):
    """
    Plot multiple datasets in one plot.

    Parameters:
        *args: Variable length argument list. Each pair of arguments represents x and y data for a plot.
        **kwargs: Additional keyword arguments for specifying plot options (e.g., color, linestyle, marker).
    """
    num_plots = len(args) // 2
    for i in range(num_plots):
        x_data = args[2*i]
        y_data = args[2*i + 1]

        # Extract plot options for this dataset
        label = kwargs.get('label', [None] * num_plots)[i]
        color = kwargs.get('color', [None] * num_plots)[i]
        linestyle = kwargs.get('linestyle', ['-'] * num_plots)[i]
        marker = kwargs.get('marker', [None] * num_plots)[i]
        x_data=x_data.reshape(-1, 1)
        y_data=y_data.reshape(-1, 1)
        plt.plot(x_data.reshape(-1, 1), y_data, label=label, linestyle=linestyle, color=color, marker=marker)

    plt.xlabel(kwargs.get('xlabel', ''))
    plt.ylabel(kwargs.get('ylabel', ''))
    plt.title(kwargs.get('title', ''))
    plt.legend()
    plt.grid(True)


# Number of samples
num_samples = 1000

# Generate input data
x1_data = np.linspace(0, 100, num_samples)
x2_data = np.exp(-0.1*x1_data) + np.random.normal(0, 0.1, num_samples)  # Add noise to sine function
x3_data = np.cos(x1_data) + np.random.normal(0, 0.1, num_samples)  # Add noise to cosine function

# Generate output data
y1_data = x3_data+np.square(x2_data) + np.random.normal(0, 0.1, num_samples)  # Add noise to squared sine function
y2_data = x1_data*np.exp(x3_data)*x2_data+ np.random.normal(0, 0.1, num_samples)  # Add noise to exponential of cosine function



# Combine input data into a single array
X_data = np.column_stack((x1_data, x2_data, x3_data))

# Combine output data into a single array
Y_data = np.column_stack((y1_data, y2_data))

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.20, shuffle=False)

# Reshape the data for LSTM input (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Plot the predictions vs. true values


# plt.figure(figsize=(4, 8))
# for i in range(2):
#     plt.subplot(1, 2, i + 1)
#     plots(X_train[:, :, 0], Y_train[:,i], X_test[:, :, 0], Y_test[:,i], xlabel='X', ylabel='Y', title='Multiple Data', label=['Training Data', 'Testing Data'], color=['red', 'blue'])


# Define the LSTM model
model = Sequential()
model.add(Input(shape=(X_train.shape[0], X_train.shape[2])))

neurons = [48, 48, 48, 48]
dropout = [0.01]

if len(neurons) != len(dropout):
    dropout = [dropout[0]] * len(neurons)

for i in range(len(neurons) - 1):
    model.add(LSTM(units=neurons[i], return_sequences=True))
    model.add(Dropout(dropout[i]))

model.add(LSTM(units=neurons[-1], return_sequences=False))  # Last LSTM layer with return_sequences=False
model.add(Dropout(dropout[-1]))

model.add(Dense(Y_train.shape[1]))  # Two output neurons for y1 and y2
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)


# Plot the predictions vs. true values
plt.figure(figsize=(4, 8))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plots(X_train[:, :, 0], Y_train[:,i], X_test[:, :, 0],Y_test[:, i],X_test[:, :, 0], predictions[:, i], label=['Training Data', 'Testing Data','Predicted Data'], color=['black', 'blue','red'])



plt.show()