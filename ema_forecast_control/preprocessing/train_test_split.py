from typing import Optional
import pandas as pd

import logging
log = logging.getLogger(__name__)

def train_test_split(df: pd.DataFrame, split_arg: Optional[int|str|pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ''' The train and test set overlap in 1 data point. It used as ground truth during 
        training and for initialization during testing. '''
    
    def determine_index_from_timestamp(df: pd.DataFrame, timestamp: pd.Timestamp) -> int|None:
        test_index = df.index[(df['DateTime']>=timestamp)]
        if len(test_index)>0:
            test_index = test_index[0]
        else:
            test_index = None
        return test_index
    
    if split_arg is None:
        return df.iloc[:], df.iloc[0:0]
    
    if isinstance(split_arg, str):
        split_arg = pd.to_datetime(split_arg)
    
    if isinstance(split_arg, pd.Timestamp):
        test_index = determine_index_from_timestamp(df, split_arg)
    else:
        test_index = split_arg

    if isinstance(test_index, int):
        df_test = df.iloc[test_index:]
        df_train = df.iloc[:test_index+1]
        log.info(f'Test index determined at position {test_index}')
    else: #None
        df_test = df.iloc[0:0]
        df_train = df.iloc[:]
        log.info(f'No test set.')
    
    return df_train, df_test