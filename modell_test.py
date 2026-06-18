import string

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import joblib
from scipy.signal import butter, filtfilt, iirnotch, stft
from scipy.stats import skew, kurtosis
import yaml
import os
from sklearn.metrics import accuracy_score

SPS = 250
with open("file_to_skip.txt", "r") as f:
    fileNumberToTest = f.read().strip()

fileNumberToTest = str(fileNumberToTest).zfill(2)

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
    freq = 60 #60Hz USA
    b, a = iirnotch(freq, quality, SPS)
    return filtfilt(b, a, data)


X_features = []
y_labels = []

window_size = int(0.8 * SPS) #window, in which blinking is analized
step_size = int(0.1 * SPS)

window_time =0
step_time = 0

y_true = []

with open("window_size.yaml", "r") as f:
    config = yaml.safe_load(f)
   
    window_size = int(config["window_size"] * SPS) #normally 0.8
    step_size = int(config["step_size"] * SPS) #normally 0.1
    
    window_time = config["window_size"]
    step_time = config["step_size"]


clf = joblib.load("blink_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("EEG-IO/S" + str(fileNumberToTest) + "_data.csv", usecols=[0, 1], sep=';')
blinks = pd.read_csv("EEG-IO/S" + str(fileNumberToTest) + "_labels.csv", skiprows=2, names=['Time (s)', 'blink'], sep=',')
blinks['Time (s)'] = pd.to_numeric(blinks['Time (s)'], errors='coerce')
blinks['blink'] = pd.to_numeric(blinks['blink'], errors='coerce')
corrupted = []

df['FP1'] = bandpass_filter(df['FP1'].values, 0.5, 70)
df['FP1'] = notch_filter(df['FP1'].values)

X_new = []
time = df['Time (s)'].values
ft_data = df['FP1'].values #filtered 
time_centers = []

#loop for sliding window
for start in range(0, len(ft_data) - window_size, step_size):
    end = start + window_size
    window = ft_data[start:end]
    window = window * np.hamming(len(window)) #norminalazation

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

        band_power(0.5, 4)
        #band_power(4, 8),
    ]
    X_new.append(features)

#check labels
y_true = []
blink_times = blinks.loc[blinks['blink'] == 1, 'Time (s)'].values
y_true = np.array([
    np.any(np.abs(blink_times - t) <= (window_time / 2))
    for t in time_centers
], dtype=int)

X_new = scaler.transform(X_new)
y_pred = clf.predict(X_new)

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)

'''
delta = bandpass_filter(df['FP1'].values, 0.5, 4)
theta = bandpass_filter(df['FP1'].values, 4, 8)

for index, row in blinks.iterrows():
    if row['blink'] == 0:
        continue
    if any(begin <= row['Time (s)'] <= end for begin, end in corrupted):
        continue
    idx = (np.abs(df['Time (s)'] - row['Time (s)'])).argmin() #closest timestamp
    aligned_time = df['Time (s)'].iloc[idx]
    plt.axvline(x=aligned_time, color='red', alpha=0.6, label='Blink' if index == 0 else "")
    #idx = (np.abs(df['Time (s)'] - row['Time (s)'])).argmin()


plt.plot(df['Time (s)'], delta, label='delta')
#plt.plot(df['Time (s)'], theta, label='theta')
plt.grid(True)

posClassified=0
for t, pred in zip(time_centers, y_pred):
    if pred ==1:
        plt.axvline(t, color='darkgreen', alpha=0.3, linewidth=2, linestyle="--")
        posClassified = posClassified +1

#print("Positively Classified: " + str(posClassified))
plt.title(f"Plot {fileNumberToTest}")
plt.legend()

# Add text at bottom
plt.figtext(0.5, 0.01, f"Winodw_size: {window_time}, step_size: {step_time}, Train accuracy: {accuracy}, Blinks Classified: {posClassified}", ha="center", fontsize=10)

####### For testing
i = 1
while os.path.exists(f"plots_LOCV/plot_{i}.pdf"):
    i += 1
fileName = f"plots_LOCV/plot_{i}.pdf"
plt.savefig(fileName)
plt.close()
# For testing

#plt.show()
'''