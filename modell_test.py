import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import joblib
from scipy.signal import butter, filtfilt, iirnotch, stft
from scipy.stats import skew, kurtosis

SPS = 250

#butterworth filter
def bandpass_filter(data, low_f, high_f, order = 4):
    nyq = SPS / 2      #Nyquisit frequency: Frequency until which can reliably be represented
    lowcut = low_f/nyq #0.5Hz highpass
    highcut = high_f/nyq #70Hz lowpass
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, data)

#optional, maybe not needed: COMMON MODE REJECTION filter or 50Hz filter for european signals
def notch_filter(data, quality=30):
    """Entfernt Netzstrom-Artefakte (50 Hz in Europa)"""
    freq = 50 #50Hz
    b, a = iirnotch(freq, quality, SPS)
    return filtfilt(b, a, data)


X_features = []
y_labels = []
window_size = int(0.8 * SPS) #window, in which blinking is analized
step_size = int(0.1 * SPS)

clf = joblib.load("blink_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("EEG-IO/S19_data.csv", usecols=[0, 1], sep=';')

df['FP1'] = bandpass_filter(df['FP1'].values, 0.5, 70)
df['FP1'] = notch_filter(df['FP1'].values)

X_new = []
time = df['Time (s)'].values
ft_data = df['FP1'].values
time_centers = []

#loop for sliding window
for start in range(0, len(ft_data) - window_size, step_size):
    end = start + window_size
    window = ft_data[start:end]
    t_center = df['Time (s)'].iloc[start:end].mean()
    time_centers.append(t_center)

    # FFT (needed for band_power)
    X = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(len(window), 1/SPS)

    def band_power(low, high):
        mask = (freqs >= low) & (freqs <= high)
        return np.sum(np.abs(X[mask])**2)

    features = [
        np.mean(window),
        np.var(window),
        np.max(window) - np.min(window),
        skew(window),
        kurtosis(window),
        band_power(0.5, 4),
        #band_power(4, 8),
    ]

    X_new.append(features)

X_new = scaler.transform(X_new)
y_pred = clf.predict(X_new)

delta = bandpass_filter(df['FP1'].values, 0.5, 4)
theta = bandpass_filter(df['FP1'].values, 4, 8)

plt.plot(df['Time (s)'], delta, label='delta')
#plt.plot(df['Time (s)'], theta, label='theta')
plt.grid(True)

for t, pred in zip(time_centers, y_pred):
    if pred ==1:
        plt.axvline(t, color='red', alpha=0.3)

plt.legend()
plt.show()