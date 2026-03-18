import pandas as pd
import matplotlib.pyplot as plt
from ema_forecast_control.preprocessing.preprocessing_functions import time_smoothing


data = pd.read_csv('ema_forecast_control/data/processed_csv_no_con/12600_12.csv')
smoothed_data = time_smoothing(data, ['EMA_mood'], kernel_width_hours=8, causal=True)
pre_smoothed_data = pd.read_csv('ema_forecast_control/data/processed_csv_no_con_smoothed_causal/12600_12.csv')

diff = (smoothed_data['Timerels'] - pre_smoothed_data['EMA_mood'])

plt.plot(diff)
plt.show()