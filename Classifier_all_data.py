import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import sys
from scipy.signal import butter, filtfilt, iirnotch, stft
from scipy.stats import skew, kurtosis
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

SPS = 250

#Butterworth filter
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

fileNumber = 0
while fileNumber <=18: #exluding file 19, to check if the ML is correct
    if fileNumber < 10: 
        filler_zero = "0"
    else:
        filler_zero = ""
    dataFile = 'EEG-IO/S' + filler_zero + str(fileNumber) + '_data.csv'
    labelsFile = 'EEG-IO/S' + filler_zero + str(fileNumber) + '_labels.csv'
    
    df = pd.read_csv(dataFile, usecols=[0, 1], sep=';')
    blinks = pd.read_csv(labelsFile)
    corrupted = []

    #if(blinks.columns[1].strip() == '0'):
        #blinks = pd.read_csv(labelsFile, skiprows=2, names=['Time (s)', 'blink'])
    if(blinks.columns[1].strip() == '1'):
        val = blinks.iloc[0].values
        corrupted.append((float(val[0]), float(val[1])))
    blinks = pd.read_csv(labelsFile, skiprows=2, names=['Time (s)', 'blink'])
    blinks['blink'] = pd.to_numeric(blinks['blink'], errors='coerce')
    blinks['Time (s)'] = pd.to_numeric(blinks['Time (s)'], errors='coerce')
    '''read file end'''

    #Filter
    df['FP1'] = bandpass_filter(df['FP1'].values, 0.5, 70)
    df['FP1'] = notch_filter(df['FP1'].values)

    #brainwaves from Banpassfilter
    delta = bandpass_filter(df['FP1'].values, 0.5, 4)
    theta = bandpass_filter(df['FP1'].values, 4, 8)
    alpha = bandpass_filter(df['FP1'].values, 8, 13)
    beta = bandpass_filter(df['FP1'].values, 13, 30)
    gamma = bandpass_filter(df['FP1'].values, 30, 100)

    #df.plot(x='Time (s)', y='FP1')

    #plt.plot(df['Time (s)'], delta, label='delta')
    #plt.plot(df['Time (s)'], theta, label='theta')
    #plt.plot(df['Time (s)'], alpha, label='alpha')
    #plt.plot(df['Time (s)'], beta, label='beta')
    #plt.plot(df['Time (s)'], gamma, label='gamma')

    def band_power(low, high):
        mask = (freqs >= low) & (freqs <= high)
        return np.sum(np.abs(X[mask])**2)

    #FFT to extract the sgnals
    ft_data = df['FP1'].values
    time = df['Time (s)'].values
    time = [float(t) for t in time]

    fft_results = []
    time_centers = []

    #loop for sliding window
    for start in range(0, len(ft_data) - window_size, step_size):
        #sliding window 
        end = start + window_size
        window = ft_data[start:end]
        window = window * np.hamming(len(window)) #norminalazation

        #skip corruted measurements
        if any(c_start <= time[start] and c_end >= time[end-1] for c_start, c_end in corrupted):
            continue

        # FFT
        X = np.fft.rfft(window)
        freqs = np.fft.rfftfreq(len(window), 1/SPS)
        
        # speichern
        fft_results.append(np.abs(X))
        
        # mean value of the window
        t_center = df['Time (s)'].iloc[start:end].mean()
        time_centers.append(t_center)
            
        # check if there is a true label in the time frame
        label = int((
            (blinks['Time (s)'] >= time[start]) &
            (blinks['Time (s)'] <= time[end-1]) &
            (blinks['blink'] == 1)
        ).any())

        y_labels.append(label)

        # Features
        features = [
            np.mean(window),
            np.var(window),
            np.max(window) - np.min(window),  # peak-to-peak
            skew(window),
            kurtosis(window),

            #relevant: delta and parts of theta waves. Since blinking is a low frequency movement
            band_power(0.5, 4),   # delta
            #band_power(4, 8),     # theta
            #band_power(8, 13),    # alpha
            #band_power(13, 30),   # beta
            #band_power(30, 70),  # gamma
        ]

        X_features.append(features)

    '''
    for index, row in blinks.iterrows():
        if row['blink'] == 0:
            continue
        if any(start <= row['Time (s)'] <= end for start, end in corrupted):
            continue
        idx = (np.abs(df['Time (s)'] - row['Time (s)'])).argmin() #closest timestamp
        aligned_time = df['Time (s)'].iloc[idx]
        plt.axvline(x=aligned_time, linestyle = "--", color='red', alpha=0.5, label='Blink' if index == 0 else "")
        #idx = (np.abs(df['Time (s)'] - row['Time (s)'])).argmin()
    '''

    fft_results = np.array(fft_results).T  
    fileNumber = fileNumber+1


#machine learning part
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=400)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(clf, "blink_model.pkl")
joblib.dump(scaler, "scaler.pkl")


