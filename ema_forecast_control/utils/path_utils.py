import os
import glob
import yaml
import numpy as np
import pandas as pd

from ema_forecast_control import ROOT
from ema_forecast_control.utils import data_utils

def model_batch_path(project_name: str) -> str:
    ''' Returns the path to the model batch folder for a given project name. '''
    batch_path = os.path.join(ROOT, 'trained_models', project_name)
    return batch_path

def join_base_path(*subpaths: str) -> str:
    ''' Joins subpaths to the base path of the ema_forecast_control package. '''
    base_path = os.path.join(ROOT, *subpaths)
    return base_path

def get_data_file(data_file_subpath: str) -> str:
    ''' Returns the full path to a data file given its subpath within the data folder. '''
    data_file = os.path.join(ROOT, 'data', data_file_subpath)
    return data_file

def read_data_file(data_file_subpath: str) -> pd.DataFrame:
    ''' Reads a CSV data file given its subpath within the data folder and returns it as a DataFrame. '''
    data_file = get_data_file(data_file_subpath)
    df = pd.read_csv(data_file)
    return df

def get_data_files(dataset_subpath: str) -> list[str]:
    ''' Returns full paths to all CSV data files within a specified dataset subpath (unsorted) '''
    data_files = glob.glob(os.path.join(ROOT, 'data', dataset_subpath, f'*.csv'))
    return data_files

def read_data_files(dataset_subpath: str) -> list[pd.DataFrame]:
    ''' Reads all CSV data files within a specified dataset subpath and returns them as a list of DataFrames, sorted by participant ID. '''
    data_files = glob.glob(os.path.join(ROOT, 'data', dataset_subpath, f'*.csv'))
    data = [pd.read_csv(file) for file in data_files]
    participants = [data_utils.determine_participant_id(df=df) for df in data]
    order = np.argsort(participants)
    data = [data[i] for i in order]
    return data

def zip_participants_data(dataset_subpath: str) -> list[str]:
    ''' Reads all CSV data files within a specified dataset subpath and returns them as a list of DataFrames, sorted by participant ID. '''
    data_files = glob.glob(os.path.join(ROOT, 'data', dataset_subpath, f'*.csv'))
    data = [pd.read_csv(file) for file in data_files]
    participants = [data_utils.determine_participant_id(df=df) for df in data]
    order = np.argsort(participants)
    participants = [participants[i] for i in order]
    data = [data[i] for i in order]
    return zip(participants, data)

def load_args(model_path: str) -> dict:
    ''' Loads training arguments from a specified model path. '''
    args_path = os.path.join(model_path, 'hypers.pkl')
    args = np.load(args_path, allow_pickle=True)
    # args = complement_args(args)
    return args

def get_project_dict(project: str) -> dict:
    ''' Loads and returns the project dictionary from a YAML file for a given project name. '''
    with open(os.path.join(ROOT, f'projects/{project}.yml'), 'r') as file:
        project_dict = yaml.safe_load(file)
    project_dict['project_name'] = project
    return project_dict

