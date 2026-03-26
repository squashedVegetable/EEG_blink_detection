import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import sys
from scipy.signal import butter, filtfilt, iirnotch, stft
from matplotlib.backends.backend_pdf import PdfPages


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

with PdfPages("all_plots.pdf") as pdf:
    filenumber = 0
    while filenumber <=19:

        if filenumber < 10: 
            filler_zero = "0"
        else:
            filler_zero = ""
        dataFile = 'EEG-IO/S' + filler_zero + str(filenumber) + '_data.csv'
        labelsFile = 'EEG-IO/S' + filler_zero + str(filenumber) + '_labels.csv'
    
        df = pd.read_csv(dataFile, usecols=[0, 1], sep=';')
        blinks = pd.read_csv(labelsFile, skiprows=2, names=['Time (s)', 'blink'], sep=',')
        blinks['Time (s)'] = pd.to_numeric(blinks['Time (s)'], errors='coerce')
        blinks['blink'] = pd.to_numeric(blinks['blink'], errors='coerce')
        corrupted = []

        if(blinks.columns[1].strip() == '0'):
            blinks = pd.read_csv(labelsFile, skiprows=2, names=['Time (s)', 'blink'])
        if(blinks.columns[1].strip() == '1'):
            val = blinks.iloc[0].values
            corrupted.append((float(val[0]), float(val[1])))
            blinks = pd.read_csv(labelsFile, skiprows=2, names=['Time (s)', 'blink'])

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
            if any(begin <= row['Time (s)'] <= end for begin, end in corrupted):
                continue
            idx = (np.abs(df['Time (s)'] - row['Time (s)'])).argmin() #closest timestamp
            aligned_time = df['Time (s)'].iloc[idx]
            plt.axvline(x=aligned_time, linestyle = "--", color='red', alpha=0.5, label='Blink' if index == 0 else "")
            #idx = (np.abs(df['Time (s)'] - row['Time (s)'])).argmin()

        plt.grid(True)
        plt.legend()
        plt.title(f"Plot {filenumber}")
        pdf.savefig()   # appends current figure as a new page
        plt.close()     # VERY IMPORTANT (avoids memory issues)
        
        filenumber = filenumber +1

        