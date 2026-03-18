from typing import Optional
import yaml
import pandas as pd
from ema_forecast_control.preprocessing import preprocessing_functions, prepare_timestamp

def get_pipeline(project: str) -> dict:
    with open(f'ema_forecast_control/projects/{project}.yml', 'r') as file:
        project = yaml.safe_load(file)
    return project['preprocessing']

def preprocessing_pipeline(df: pd.DataFrame, pipeline: Optional[dict]) -> pd.DataFrame:
    if pipeline is not None:
        df = df.copy(deep=True)
        for function in pipeline:
            if isinstance(function, str):
                f_name = function
                f_kwargs = {}
            elif isinstance(function, dict):
                f_name = next(iter(function))
                f_kwargs = function[f_name]
            df = getattr(preprocessing_functions, f_name)(df, **f_kwargs)

    return df