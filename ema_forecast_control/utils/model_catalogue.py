import os
from joblib import Parallel, delayed
import hashlib
import yaml
import glob
import pandas as pd
import numpy as np

from ema_forecast_control.utils import path_utils, eval_utils

class ModelCatalogue:

    def __init__(self, project_name: str, num_workers: int=1, force_new_catalogue: bool=False, **hyperparameters):
        self.main_dir = path_utils.join_base_path('trained_models', project_name)
        self.hyperparameters = hyperparameters
        catalogue_loaded = self._load_catalogue_if_exists()
        if not catalogue_loaded or force_new_catalogue:
            if len(hyperparameters)==0:
                model_dirs = self._collect_model_dirs()
            else:
                model_dirs = self._filter_model_dirs_by_hyperparameters(**hyperparameters)
            catalogue = []
            def process_model_dir(d):
                args = path_utils.load_args(d)
                props = pd.DataFrame(index=[0])
                props['model_dir'] = d
                props['participant'] = str(args['participant'])
                if props['model_timestep'] is not None:
                    props['model_timestep'] = int(float(args['train_test_split']))
                else:
                    props['model_timestep'] = np.inf
                props['model_datetime'] = 'NotImplemented'
                props['train_on_last_n_steps'] = args['train_on_last_n_steps']
                return props
            catalogue = Parallel(n_jobs=num_workers)(delayed(process_model_dir)(d) for d in model_dirs)
            self.catalogue = pd.concat(catalogue, ignore_index=True)
            self.catalogue = self.catalogue.sort_values('model_dir')
            self._save_catalogue()


    def _save_catalogue(self):
        hypers_as_yaml = yaml.dump(self.hyperparameters, sort_keys=True)
        hypers_hash = hashlib.sha256(hypers_as_yaml.encode()).hexdigest()[:6]
        path = path_utils.join_base_path(self.main_dir, 'model_catalogues', hypers_hash)
        os.makedirs(path, exist_ok=True)        
        with open(os.path.join(path, 'hyperparameters.yml'), 'w') as file:
            yaml.safe_dump(self.hyperparameters, file, sort_keys=True)
        self.catalogue.to_csv(os.path.join(path, 'model_catalogue.csv'))

    def _load_catalogue_if_exists(self):
        hypers_as_yaml = yaml.dump(self.hyperparameters, sort_keys=True)
        hypers_hash = hashlib.sha256(hypers_as_yaml.encode()).hexdigest()[:6]
        catalogue_path = path_utils.join_base_path(self.main_dir, 'model_catalogues', hypers_hash, 'model_catalogue.csv')
        if os.path.exists(catalogue_path):
            self.catalogue = pd.read_csv(catalogue_path, index_col=0)
            self.catalogue['participant'] = self.catalogue['participant'].astype('str')
            return True
        else:
            return False

    def _collect_model_dirs(self) -> list:
        model_paths = glob.glob(os.path.join(self.main_dir, '**/model*.pt'), recursive=True)
        return sorted(set([os.path.split(p)[0] for p in model_paths]))

    def _filter_model_dirs_by_hyperparameters(self, **params) -> list:
        model_dirs = self._collect_model_dirs()
        filtered_dirs = []
        for d in model_dirs:
            args = path_utils.load_args(d)
            include = True
            for p in params:
                if args[p]!=params[p]:
                    include = False
            if include:
                filtered_dirs.append(d)
        return filtered_dirs

    def get_all_model_dirs(self, participant: str):
        models = self.catalogue[self.catalogue['participant']==participant].sort_values('model_timestep')
        choose_dirs = models['model_dir'].to_list()
        return choose_dirs

    def get_latest_model_dirs(self, participant: str, timestep: int|None=None, datetime: int|None=None) -> list[str]:
        if datetime is not None:
            raise NotImplementedError('Choose model dir by datetime')
        models = self.catalogue[self.catalogue['participant']==participant].sort_values('model_timestep')
        if timestep is None:
            models = models.loc[models['model_timestep']==models['model_timestep'].max()]
            choose_dirs = models['model_dir'].to_list()
        elif (models['model_timestep'] < timestep).any():
            models = models.loc[models['model_timestep']<timestep]
            models = models.loc[models['model_timestep']==models['model_timestep'].max()]
            choose_dirs = models['model_dir'].to_list()
        else:
            choose_dirs = []
        return choose_dirs
    
    def get_best_latest_model_dir(self, participant: str, timestep: int|None=None, datetime: int|None=None):
        model_dirs = self.get_latest_model_dirs(participant, timestep, datetime)
        if len(model_dirs) > 0:
            if len(model_dirs)>1:
                run_dirs = set([os.path.dirname(m) for m in model_dirs])
                if len(run_dirs)==1:
                    run_dir = list(run_dirs)[0]
                    best_run = eval_utils.determine_best_run(run_dir)
                    model_dir = os.path.join(run_dir, best_run)
                else:
                    raise RuntimeError("In latest model dirs, there seem to be more than 1 model configuration. You can pick a specific configuration by specifying hyperparameters in ModelCatalogue.")
            else:
                model_dir = model_dirs[0]
        else:
            model_dir = None
        return model_dir

    def get_best_model_dirs(self, participant: str):
        models = self.catalogue.loc[self.catalogue['participant']==participant].sort_values('model_timestep')
        dirs = []
        for timestep in models['model_timestep'].unique():
            current_models = models.loc[models['model_timestep']==timestep]
            if len(current_models)>1:
                run_dirs = set([os.path.dirname(m) for m in current_models['model_dir']])
                if len(run_dirs)==1:
                    run_dir = list(run_dirs)[0]
                    best_run = eval_utils.determine_best_run(run_dir)
                    model_dir = os.path.join(run_dir, best_run)
                else:
                    raise RuntimeError("In latest model dirs, there seem to be more than 1 model configuration. You can pick a specific configuration by specifying hyperparameters in ModelCatalogue.")
            else:
                model_dir = current_models['model_dir'].values[0]
            dirs.append(model_dir)
        return dirs
    

