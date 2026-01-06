from typing import Optional
import pandas as pd

def convert_to_relative_time(timestamp: pd.Timestamp, time_anchor: pd.Timestamp) -> pd.Series:
    return (timestamp - time_anchor).dt.total_seconds()

def prepare_timestamp(df: pd.DataFrame, absolute_datetime_column: Optional[str]=None, 
                        relative_datetime_column: Optional[str]=None, time_anchor: str='2022-01-01 00:00:00',
                        datetime_format: Optional[str]='%Y-%m-%d %H:%M:%S') -> pd.DataFrame:

    time_anchor = pd.to_datetime(time_anchor, format=datetime_format)
    if absolute_datetime_column is not None:
        datetime_series = pd.to_datetime(df[absolute_datetime_column], format=datetime_format)
    elif relative_datetime_column is not None:
        datetime_series = time_anchor + pd.to_timedelta(df[relative_datetime_column], unit='s')
    else:
        raise RuntimeError('Either absolute_datetime_column or relative_datetime_column and time_anchor must be provided.')
           
    df['Date'] = datetime_series.dt.normalize()
    df['Time'] = datetime_series.dt.time
    df['DateTime'] = datetime_series
    df['DayNr'] = (df['Date'] - df['Date'].min()).dt.days
    df['Timerels'] = convert_to_relative_time(datetime_series, time_anchor)

    df = df.sort_values('Timerels').reset_index(drop=True)
    df.index.name = 'Timesteps'

    return df