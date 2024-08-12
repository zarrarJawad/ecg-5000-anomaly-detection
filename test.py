import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from darts.models import TCNModel
import torch
import random
from darts.ad.anomaly_model import ForecastingAnomalyModel
from darts.ad.scorers import NormScorer

# Load the ECG train and test data
ecg_train = pd.read_csv(r'data\ECG5000_TRAIN.txt', sep='\s+')
ecg_test = pd.read_csv(r'data\ECG5000_TEST.txt', sep='\s+')

# Merge the train and test data into a single DataFrame
ecg_combined = pd.concat([ecg_train, ecg_test], ignore_index=True)

# Define normal and anomaly data
normal_data = ecg_combined[ecg_combined.iloc[:, 0] == 1]
anomaly_data = ecg_combined[ecg_combined.iloc[:, 0] != 1]

# Handle NaN values by filling them with the column mean
normal_data_filled = normal_data.fillna(normal_data.mean())
anomaly_data_filled = anomaly_data.fillna(anomaly_data.mean())

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
normal_ecg_data = scaler.fit_transform(normal_data_filled.iloc[:, 1:])
anomaly_ecg_data = scaler.transform(anomaly_data_filled.iloc[:, 1:])

# Split normal data into 80% train, 15% validation, 5% test
train_data, temp_data = train_test_split(normal_ecg_data, test_size=0.20, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.25, random_state=42)

train_series = TimeSeries.from_values(train_data)
validation_series = TimeSeries.from_values(validation_data)
test_series = TimeSeries.from_values(test_data)

# Load the saved model
model = TCNModel.load("tcn_model.pth")

# Ensure the correct shape for the input data
def ensure_correct_shape(series, expected_width):
    if series.width != expected_width:
        print(f"Adjusting series width from {series.width} to {expected_width}.")
        return TimeSeries.from_values(np.repeat(series.values(), expected_width, axis=1))
    return series

# Create the TimeSeries object for a random anomalous ECG sample
anomalous_ecg_series = TimeSeries.from_values(anomaly_ecg_data[random.randint(0, len(anomaly_ecg_data) - 1), :].reshape(-1, 1))
anomalous_ecg_series = ensure_correct_shape(anomalous_ecg_series, model.output_dim)

# Create the TimeSeries object for a random normal ECG sample
normal_ecg_series = TimeSeries.from_values(normal_ecg_data[random.randint(0, len(normal_ecg_data) - 1), :].reshape(-1, 1))
normal_ecg_series = ensure_correct_shape(normal_ecg_series, model.output_dim)

# Fit the anomaly detection model
anomaly_model = ForecastingAnomalyModel(
    model=model,
    scorer=NormScorer()
)

# Calculate anomaly scores for the anomalous ECG
anomaly_scores_ts = anomaly_model.score(anomalous_ecg_series)
anomaly_scores = anomaly_scores_ts.values().flatten()

# Determine threshold for anomaly detection
threshold = np.percentile(anomaly_scores, 80)
anomalous_chunks = anomaly_scores > threshold

# Plotting function for the ECG signal with highlighted anomalies
def plot_ecg_with_highlights(ecg_series, anomaly_scores, anomalous_chunks, title="ECG Signal with Anomalies"):
    plt.figure(figsize=(14, 7))
    x_values = np.arange(len(ecg_series.values().flatten()))
    y_values = ecg_series.values().flatten()

    plt.plot(x_values, y_values, color='green', label='Normal ECG Signal')

    # Highlight the anomalous segments in red
    for i, is_anomalous in enumerate(anomalous_chunks):
        if is_anomalous:
            start_index = i * len(y_values) // len(anomalous_chunks)
            end_index = min(start_index + len(y_values) // len(anomalous_chunks), len(x_values))
            plt.plot(x_values[start_index:end_index], y_values[start_index:end_index], color='red')

    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('ECG Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the normal ECG
plot_ecg_with_highlights(normal_ecg_series, [], [], title='Normal ECG Signal')

# Plot the anomalous ECG with detected anomalies
plot_ecg_with_highlights(anomalous_ecg_series, anomaly_scores, anomalous_chunks, title='Anomalous ECG Signal')
