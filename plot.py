import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

fileNumber = 0

fileNumber = str(fileNumber).zfill(2)

dataFile = 'EEG-IO/S' + fileNumber + '_data.csv'
labelsFile = 'EEG-IO/S' + fileNumber + '_labels.csv'

df = pd.read_csv(dataFile, usecols=[0, 1], sep=';')
blinks = pd.read_csv(labelsFile, skiprows=2, usecols=[0])

df.plot(x='Time (s)', y='FP1')
for index, row in blinks.iterrows():
    plt.axvline(x=row['Time (s)'], color='red', linestyle='--', alpha=0.7, label='Blink' if row['Time (s)'] == blinks['Time (s)'].iloc[0] else "")
    idx = (np.abs(df['Time (s)'] - row['corrupt'])).argmin()

plt.show()