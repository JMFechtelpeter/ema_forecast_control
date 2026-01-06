import os
import glob
import re
import numpy as np
import pandas as pd

from ema_forecast_control import ROOT

def get_data_file(data_file_subpath: str) -> str:
    data_file = os.path.join(ROOT, 'data', data_file_subpath)
    return data_file

def read_data_file(data_file_subpath: str) -> pd.DataFrame:
    data_file = get_data_file(data_file_subpath)
    df = pd.read_csv(data_file)
    return df

def get_data_files(dataset_subpath: str) -> list[str]:
    data_files = glob.glob(os.path.join(ROOT, 'data', dataset_subpath, f'*.csv'))
    participants = np.array([int(re.search(fr'\_([0-9]+).csv', file).group(1)) for file in data_files])
    order = participants.argsort()
    data_files = [data_files[i] for i in order]
    return data_files