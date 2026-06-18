import pandas as pd
import numpy as np
import joblib
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
import yaml

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score
)

SPS = 250

# -----------------------
# Filters
# -----------------------
def bandpass_filter(data, low_f, high_f, order=4):
    nyq = SPS / 2
    b, a = butter(order, [low_f/nyq, high_f/nyq], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, quality=30):
    freq = 50
    b, a = iirnotch(freq, quality, SPS)
    return filtfilt(b, a, data)

# -----------------------
# Load config
# -----------------------
with open("window_size.yaml", "r") as f:
    config = yaml.safe_load(f)
    window_size = int(config["window_size"] * SPS)
    step_size = int(config["step_size"] * SPS)
    window_time = config["window_size"]

# -----------------------
# Load model
# -----------------------
clf = joblib.load("blink_model copy.pkl")
scaler = joblib.load("scaler copy.pkl")

# -----------------------
# Storage
# -----------------------
accs, f1s, aucs = [], [], []

# -----------------------
# Loop over all files
# -----------------------
for fileNumber in range(1, 12):

    if fileNumber == 5:
        continue

    filler_zero = "0" if fileNumber < 10 else ""

    dataFile = f"Our_data_classify/ADS1291_{filler_zero}{fileNumber}.csv"
    labelsFile = f"Our_data_classify/events_{filler_zero}{fileNumber}.csv"

    print(f"\nProcessing file {fileNumber}")

    df = pd.read_csv(dataFile, comment="#", sep=",", skipinitialspace=True)
    events = pd.read_csv(labelsFile, comment="#")

    blink_times = np.array(events["event_elapsed_ms"] / 1000.0)

    raw = df["ADS1291_EXG"].astype(float).values
    raw = raw - np.mean(raw)
    raw = bandpass_filter(raw, 0.5, 70)
    raw = notch_filter(raw)

    time = df["sample_index"].values / SPS

    # -----------------------
    # Feature extraction
    # -----------------------
    X_new = []
    time_centers = []

    for start in range(0, len(raw) - window_size, step_size):
        end = start + window_size

        window = raw[start:end]
        window_hamming = window * np.hamming(len(window))

        t_start = time[start]
        t_end = time[end - 1]

        time_centers.append((t_start + t_end) / 2)

        X = np.fft.rfft(window_hamming)
        freqs = np.fft.rfftfreq(len(window_hamming), 1 / SPS)

        def band_power(low, high):
            mask = (freqs >= low) & (freqs <= high)
            return np.sum(np.abs(X[mask]) ** 2)

        features = [
            np.mean(window),
            np.var(window),
            np.max(window) - np.min(window),
            skew(window),
            kurtosis(window),
            band_power(0.5, 4),
        ]

        X_new.append(features)

    X_new = np.array(X_new)

    # -----------------------
    # Ground truth
    # -----------------------
    y_true = np.array([
        np.any(np.abs(blink_times - t) <= (window_time / 2))
        for t in time_centers
    ], dtype=int)

    # -----------------------
    # Prediction
    # -----------------------
    X_new = scaler.transform(X_new)
    y_pred = clf.predict(X_new)

    # probabilities for AUC
    y_prob = clf.predict_proba(X_new)[:, 1]

    # -----------------------
    # Metrics
    # -----------------------
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    accs.append(acc)
    f1s.append(f1)
    aucs.append(auc)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"AUC      : {auc:.4f}")

# -----------------------
# Summary
# -----------------------
print("\n======================")
print("FINAL RESULTS")
print("======================")
print(f"Mean Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Mean F1-score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Mean AUC      : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")