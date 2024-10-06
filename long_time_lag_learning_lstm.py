import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Generating some dummy sequential data
def generate_data(sequence_length=100, num_sequences=1000, time_lag=10):
    X = []
    y = []
    for _ in range(num_sequences):
        sequence = np.sin(np.linspace(0, 50, sequence_length))  # Sinusoidal data
        X.append(sequence[:-time_lag])  # Input data excluding the lag
        y.append(sequence[time_lag:])   # Output data (shifted by time_lag)
    return np.array(X), np.array(y)

# Data preprocessing
sequence_length = 100
X, y = generate_data(sequence_length=sequence_length, num_sequences=1000, time_lag=10)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# Reshaping data to fit LSTM [samples, timesteps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
y_scaled = y_scaled.reshape((y_scaled.shape[0], y_scaled.shape[1], 1))

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length - 10, 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dense(units=1))  # Output layer

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_scaled, y_scaled, epochs=20, batch_size=64, verbose=2)

# Predicting with the model
predicted_sequences = model.predict(X_scaled)

# Inversing the scaling to get back to the original scale
predicted_sequences_rescaled = scaler.inverse_transform(predicted_sequences.reshape(-1, sequence_length - 10))

# Visualizing the predictions vs actual data
import matplotlib.pyplot as plt

# Plotting one example of prediction vs actual
plt.plot(y[0], label='Actual')
plt.plot(predicted_sequences_rescaled[0], label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Time Series')
plt.show()
