import logging
from typing import Iterable, Optional
import pandas as pd
import numpy as np

# log = logging.getLogger(__name__)

    
class helpers:

    @staticmethod
    def determine_participant(df, participant_column: str='Participant') -> int:
        participant = df[participant_column].unique()
        if len(participant)>1:
            raise RuntimeError('Participant could not be uniquely determined')
        if not isinstance(participant.item(), int):
            raise RuntimeError('Participant id is not an integer')
        return participant.item()
    

    @staticmethod
    def ensure_features_exist(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
        missing_features = set(feature_names).difference(df.columns)
        if len(missing_features) > 0:
            # log.warning('Features %s are not present in the dataset and will be filled with nans', str(missing_features))
            df[list(missing_features)] = np.nan
        return df
    
    @staticmethod
    def normalize(df: pd.DataFrame, feature_names: Iterable[str], to_interval=[-1,1], origin=[1,7]) -> pd.DataFrame:
        """ Linear normalzation """
        a = (to_interval[0] - to_interval[1]) / (origin[0] - origin[1])
        b = to_interval[1] - a * origin[1]
        df[feature_names] = a*df[feature_names] + b
        return df
    
    @staticmethod
    def nanconvolve(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Equivalent to numpy.convolve(array, kernel, mode='valid') except:
            - kernel is divided by kernel sum
            - if there are nans in the convolution, the dot product is taken only across the notnan values.
                only if there is a sequence of nans as long as the kernel, nans will remain in the data.
        """
        array_without_nan = np.nan_to_num(array, nan=0.)
        M = len(kernel)
        N = len(array)
        res = np.zeros(N-M+1)            
        for i in range(N-M+1):
            local_notnans = ~np.isnan(array[i:i+M])
            local_kernel = kernel[local_notnans]
            if local_kernel.sum() != 0:            
                res[i] = np.dot(array_without_nan[i:i+M], kernel/(kernel[local_notnans].sum()))
            else:
                res[i] = np.nan
        return res
    
    @staticmethod
    def gaussian_kernels(x: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
        ''' returns gaussian pdfs with means <means> and std <std> evaluated at <x>
            Output format: (len(means), len(x))'''
        x = np.array(x)[np.newaxis, :]
        mean = np.array(mean)[:, np.newaxis]
        exponent = - (x - mean)**2 / (2*std**2)
        res = np.exp(exponent) / (np.sqrt(2*np.pi)*std)
        return res.squeeze()





def consistent_time_bin_number(df: pd.DataFrame, min_time_bins_per_day: int=6, ignore_empty_days: bool=False, 
                               time_anchor: Optional[str]='2022-01-01 00:00:00',
                               start_time: Optional[str]='00:00:00', end_time: Optional[str]='23:59:59') -> pd.DataFrame:
    """
    Provides a minimum number of time bins per day for the dataset, filling all additional bins with nans.
    threshold_seconds: if there are gaps larger than this, new bins are inserted in there, else, at the beginning or end of the day.
    """    

    def create_empty_df(columns: Iterable, timerels: np.ndarray, date: pd.Timestamp, day_nr: int):
        edf = pd.DataFrame(columns=columns)
        edf['Participant'] = participant
        edf['Date'] = date
        edf['Time'] = edf['DateTime'].dt.time
        edf['DateTime'] = (pd.to_timedelta(timerels, unit='s') + pd.Timestamp(time_anchor))
        edf['DayNr'] = day_nr
        edf['Timerels'] = timerels
        return edf
    
    def insert_empty_rows_into_df(df: pd.DataFrame, timerels: np.ndarray, date: pd.Timestamp, day_nr: int):
        edf = create_empty_df(df.columns, timerels, date, day_nr)
        new_df =  pd.concat([df, edf], ignore_index=True)
        new_df.sort_values(by='Timerels', inplace=True, ignore_index=True)
        return new_df
    
    def insert_empty_rows_optimally(df: pd.DataFrame, min: int, max: int, date: pd.Timestamp, day_nr: int) -> pd.DataFrame:
        timerels = df['Timerels']
        dtype = int
        array = timerels.to_numpy(dtype=float)
        if len(array)==0 or array[0] > min:
            array = np.r_[min, array]
            to_length += 1
        if array[-1] < max:
            array = np.r_[array, max]
            to_length += 1
        gaps = np.diff(array)
        gaps_after_insertion = gaps*1
        insert_into_gaps = np.zeros_like(gaps, dtype=int)
        for k in range(to_length-len(array)):
            gap_index = gaps_after_insertion.argmax()
            insert_into_gaps[gap_index] += 1
            gaps_after_insertion[gap_index] = gaps[gap_index] / (insert_into_gaps[gap_index] + 1)
        for i in range(len(gaps)-1, -1, -1):
            if insert_into_gaps[i] > 0:
                insertion_timerels = np.linspace(array[i], array[i+1], insert_into_gaps[i]+2, dtype=dtype)[1:-1]
                df = insert_empty_rows_into_df(df, insertion_timerels, date, day_nr)
        return df
    
    single_date_dfs = []
    last_timerel = df['Timerels'].iloc[-1]
    participant = helpers.determine_participant(df)
    for datetime in pd.date_range(df['Date'].min(), df['Date'].max()):
        date = datetime.normalize()
        day_nr = (date - df['Date'].min()).days
        date_df = df[df['Date']==date]
        date_startrel = (datetime + pd.Timedelta(start_time) - time_anchor).total_seconds()
        date_endrel = (datetime + pd.Timedelta(end_time) - time_anchor).total_seconds()
        if not (len(date_df)==0 and ignore_empty_days):
            date_df = insert_empty_rows_optimally(date_df, date_startrel, date_endrel, date, day_nr)
        single_date_dfs.append(date_df)
            
    df = pd.concat(single_date_dfs, ignore_index=True)
    df = df[df['Timerels']<=last_timerel]    #it makes no sense to add empty bins on the last day, so drop them

    return df

def time_smoothing(df: pd.DataFrame, columns_to_smooth: Iterable[str], 
                   kernel_std_hours: float=1.5, kernel_width_hours: float=8.0, 
                   causal: bool=True):    
        
    evaluate_at = df['Timerels'].to_numpy(dtype=int)
    new_df = df.copy()
    
    kernel_std_sec = kernel_std_hours * 3600
    weights = helpers.gaussian_kernels(df['Timerels'], evaluate_at, kernel_std_sec) * 3600
    # weights are of shape (evaluation_time * data_row)
    if kernel_width_hours is not None:
        diff_hours = np.abs((evaluate_at[:, np.newaxis] - df['Timerels'].to_numpy(dtype=int)[np.newaxis, :])) / 3600
        weights[diff_hours > kernel_width_hours] = 0.
    if causal:
        weights[np.triu_indices_from(weights, 1)] = 0.
    for f in columns_to_smooth:
        f_data = df[f].to_numpy()
        f_weights = weights
        f_weights[:, np.isnan(f_data)] = 0.
        f_data = np.nan_to_num(f_data, nan=0.)
        weights_sum = f_weights.sum(axis=1, keepdims=True)
        weights_sum[weights_sum==0] = np.nan
        weights = weights / weights_sum
        convolved_data = np.einsum('r,tr->t', f_data, weights)
        new_df[f] = convolved_data
    return new_df

def zero_impute_input(df: pd.DataFrame, input_columns: Iterable[str]) -> pd.DataFrame:
    df[input_columns] = df[input_columns].fillna(value=0)
    return df

def feature_selection(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    return df[list(feature_names)]