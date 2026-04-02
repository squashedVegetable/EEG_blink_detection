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
import matplotlib.ticker as ticker

SPS = 250
window_size = int(0.5 * SPS) #window, in which blinking is analized
step_size = int(0.1 * SPS)

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
    freq = 60 #50Hz europe, 60Hz USA
    b, a = iirnotch(freq, quality, SPS)
    return filtfilt(b, a, data)

'''read file'''
if len(sys.argv) < 2 or not sys.argv[1]:
    fileNumber = int(input("File number: "))
else:
    fileNumber = sys.argv[1]

timeStampStart = float(input("start where?"))
while timeStampStart == "" or timeStampStart == " ":
    timeStampStart = float(input("Until where?: "))

timeStampEnd = float(input("Until where?: "))
while timeStampEnd == "" or timeStampEnd == " ":
    timeStampEnd = float(input("Until where?: "))

fileNumber = str(fileNumber).zfill(2)

dataFile = 'EEG-IO/S' + fileNumber + '_data.csv'
labelsFile = 'EEG-IO/S' + fileNumber + '_labels.csv'

df = pd.read_csv(dataFile, usecols=[0, 1], sep=';')
blinks = pd.read_csv(labelsFile)
corrupted = []

timeDataStart = df["Time (s)"] >= timeStampStart  # rows >= start
if timeDataStart.any():
    startTimeIndex = np.flatnonzero(timeDataStart)[0]  # first True
else:
    startTimeIndex = 0 

timeData = df["Time (s)"] <= timeStampEnd
if timeData.any():
    endTimeIndex = np.flatnonzero(timeData)[-1]  
    print("Floor index:", endTimeIndex)

df = df.iloc[startTimeIndex:endTimeIndex+1]
lastTime = df["Time (s)"].iloc[-1] 
firstTime = df["Time (s)"].iloc[0]

if(blinks.columns[1].strip() == '0'):
    blinks = pd.read_csv(labelsFile, skiprows=2, names=['Time (s)', 'blink'])
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

plt.plot(df['Time (s)'], delta, label='delta')
plt.plot(df['Time (s)'], theta, label='theta')
#plt.plot(df['Time (s)'], alpha, label='alpha')
#plt.plot(df['Time (s)'], beta, label='beta')
#plt.plot(df['Time (s)'], gamma, label='gamma')

for index, row in blinks.iterrows():
    if row['blink'] == 0:
        continue
    if any(start <= row['Time (s)'] <= end for start, end in corrupted):
        continue
    if row['Time (s)'] < firstTime or row['Time (s)'] > lastTime:
        continue
    idx = (np.abs(df['Time (s)'] - row['Time (s)'])).argmin() #closest timeStampEnd
    aligned_time = df['Time (s)'].iloc[idx]
    if aligned_time == lastTime:
        continue
    plt.axvline(x=aligned_time, linestyle = "--", color='red', alpha=0.5, label='Blink' if index == 0 else "")
    #idx = (np.abs(df['Time (s)'] - row['Time (s)'])).argmin()

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
for i, label in enumerate(ax.xaxis.get_ticklabels()):
    if i % 2 != 0:
        label.set_visible(False)
plt.grid(True)
plt.legend()
plt.show()

