from collections.abc import Iterable
from typing import Optional
import pandas as pd
import numpy as np

def rescale_data(df, columns: Iterable[str], interval: Iterable):
    """
    Rescales negatively phrased ema items, such that 7 ist the best and 1 is the worst value
    """
    assert len(interval)==2, "Interval must be of length 2"
    def rescale(ema):
        return interval[1] - (ema - interval[0])
    df[columns] = df[columns].apply(rescale)
    return df

def zscore(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for f in columns:
        data = df[f].to_numpy(dtype=float)
        if (~np.isnan(data)).sum()>0:
            data -= np.nanmean(data)
            if np.nanstd(data)>0:
                data /= np.nanstd(data)
            df[f] = data
    return df 

def ffill_bfill(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df[columns] = df[columns].fillna(method='ffill', axis=0)
    df[columns] = df[columns].fillna(method='bfill', axis=0)
    return df

def determine_participant_id(df: Optional[pd.DataFrame]=None, data_path: Optional[str]=None) -> int:
    if df is None and data_path is not None:
        df = pd.read_csv(data_path)
    if 'participant' in df.columns:
        return str(df['participant'].iloc[0].item())
    elif 'Participant' in df.columns:
        return str(df['Participant'].iloc[0].item())
    else:
        raise ValueError('No participant column found in the data.')