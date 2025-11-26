import os
import sys
sys.path.append('/home/janik.fechtelpeter/Documents/bptt/')
import glob
import re
import itertools as it
import pandas as pd
from tqdm import tqdm
import torch as tc
import numpy as np
import matplotlib.pyplot as plt
import eval_reallabor_utils
import data_utils
from numpy.typing import ArrayLike

ema_names = ['EMA_mood','EMA_disappointed','EMA_scared','EMA_worry',
    'EMA_down','EMA_sad','EMA_confidence','EMA_stress','EMA_lonely',
    'EMA_energetic','EMA_concentration','EMA_resilience','EMA_tired',
    'EMA_satisfied', 'EMA_relaxed']
intervention_names = ['interactive1', 'interactive2', 'interactive3', 'interactive4',
                          'interactive5', 'interactive6','interactive7', 'interactive8']

class EffectsPrediction:

    def __init__(self, data_dir, model_dir, model_for_prediction: str, try_to_load_predictions_file: bool=True):
        self.data_dir = os.path.split(data_dir)[1]
        self.model_dir = os.path.split(model_dir)[1]
        self.data = self.load_data(data_dir)
        self.participants = self.data.index.get_level_values('participant').unique()
        self.model_for_prediction = model_for_prediction
        if try_to_load_predictions_file:
            predictions_file_path = data_utils.join_base_path(f'bptt/eval_reallabor/predictions/{model_for_prediction}__{self.data_dir}__{self.model_dir}.csv')
            try:
                self.predicted = pd.read_csv(predictions_file_path, index_col=[0,1])
            except:
                try_to_load_predictions_file = False
        if not try_to_load_predictions_file:
            if model_for_prediction=='current':
                self.predicted = self.predict_with_current_models(model_dir)
            elif model_for_prediction=='latest':
                self.predicted = self.predict_with_latest_models(model_dir)
            else:
                raise ValueError('specify model for prediction')
            self.predicted.to_csv(os.path.join('predictions', f'{model_for_prediction}__{self.data_dir}__{self.model_dir}.csv'))
    
    def plot_simulated_against_data(self, participant: int|list[int], time_limits: list=None, features: str|list=ema_names, from_first_prediction: bool=False,
                                    data_plot_kwargs: dict={}, simulated_plot_kwargs: dict={}, model_change_plot_kwargs: dict={},
                                    legend: bool=False):
        if isinstance(participant, int):
            participant = [participant]
        if isinstance(features, str):
            features = [features]
        gt = self.data.loc[participant]
        pred = self.predicted.loc[participant]
        if time_limits is not None:
            time_mask = (gt.index.get_level_values('timesteps') >= time_limits[0]) & (gt.index.get_level_values('timesteps') <= time_limits[1])
        else:
            time_mask = np.ones(len(gt), dtype=bool)
        if from_first_prediction:
            new_model_mask = time_mask & ((pred['model_id'] != pred['model_id'].shift(1)).values)
            time_mask = time_mask & (pred[ema_names].notna().all(axis=1).values)
        n_feat = len(features)
        n_participants = len(participant)
        fig, axes = plt.subplots(n_feat*n_participants, 1, figsize=(8, n_feat*n_participants*2 + 1), squeeze=False)
        data_plot_kwargs = {'marker':'.', **data_plot_kwargs}
        simulated_plot_kwargs = {'marker':'o', 'fillstyle': 'none', **simulated_plot_kwargs}
        model_change_plot_kwargs = {'marker':'s', 'fillstyle': 'full', 'linestyle':'', **model_change_plot_kwargs}
        failed_participants = []
        idx = pd.IndexSlice
        for (p, f), ax in zip(it.product(participant, features), axes.flat):
            try:
                x_axis = gt.loc[(p, time_mask), f].dropna().index.get_level_values('timesteps')
                new_model_idx = gt.loc[(p, new_model_mask), f].index.get_level_values('timesteps')
            except KeyError as e:
                failed_participants.append(p)
                continue
            gt_line, = ax.plot(x_axis, gt.loc[idx[p, x_axis], f].values, **data_plot_kwargs)
            pred_line, = ax.plot(x_axis, pred.loc[idx[p, x_axis], f].values, **simulated_plot_kwargs)
            nm_line, = ax.plot(new_model_idx, pred.loc[idx[p, new_model_idx], f].values, **model_change_plot_kwargs)
            ax.set(ylim=(0.5, 7.5), yticks=np.arange(1,8), yticklabels=['1', '', '', '4', '', '', '7'], title=f'participant {p}, {f}')
            if legend:
                ax.legend([gt_line, pred_line, nm_line], ['ground truth', 'prediction', 'model update'],
                          loc='center left', bbox_to_anchor=(1., 0.5))
        # if len(failed_participants)>0:
        #     print(f'No data to plot from participants {failed_participants}.')
        return axes


    def load_data(self, data_dir):
        data_files = glob.glob(os.path.join(data_dir, '*.csv'))
        all_data = []
        for file in data_files:
            participant = int(re.search('_([0-9]+)\.csv', file).group(1))
            data = pd.read_csv(file, index_col=0)[['DateTime'] + ema_names + intervention_names]
            data.index = pd.MultiIndex.from_product([[participant], data.index], 
                                                                names=['participant', 'timesteps'])
            all_data.append(data)
        all_data = pd.concat(all_data, ignore_index=False, axis=0)
        return all_data.sort_index()
    
    def calc_effects(self, use_predicted: bool=False, EMI_only: bool=False):
        if use_predicted:
            df = self.predicted
        else:
            df = self.data
        idx = pd.IndexSlice
        diff = df.copy(deep=True)
        diff[ema_names] = df[ema_names].diff()
        diff.loc[idx[:,0], ema_names] = np.nan
        diff[ema_names] = diff[ema_names].shift(-1)
        # diff[intervention_names] = df[intervention_names]
        if EMI_only:
            diff = diff.loc[self.data[intervention_names].sum(axis=1)>0]
        return diff
    
    def predict_with_current_models(self, experiment_dir: str):
        predicted_data = pd.DataFrame(index=self.data.index, columns=self.data.columns)
        predicted_data[['model_timestep', 'model_id']] = np.nan
        print('Making predictions...')
        model_catalogue = eval_reallabor_utils.ModelCatalogue(experiment_dir)
        for participant in tqdm(self.participants):
            participant_data = self.data.loc[participant]
            current_dir = None
            for timestep in participant_data.index[5:-1]:
                # d = eval_reallabor_utils.get_current_model_dir(experiment_dir, participant, timestep)
                d = model_catalogue.get_current_model_dir(participant, timestep=timestep)
                if d is None:
                    continue
                elif d != current_dir:
                    current_dir = d
                    model, _ = eval_reallabor_utils.load_model_and_data(current_dir)
                emas = tc.tensor(participant_data.loc[timestep-4:timestep, ema_names].values).float() 
                inputs = tc.tensor(participant_data.loc[timestep-4:timestep, intervention_names].values).float() 
                predicted_traj, _ = model.generate_free_trajectory(emas[-1:], inputs[-1:], 2,
                                                                    prewarm_data=emas[:-1], 
                                                                    prewarm_inputs=inputs[:-1]) 
                predicted_data.loc[(participant, timestep+1), ema_names] = predicted_traj[1].numpy()
                predicted_data.loc[(participant, timestep+1), 'model_timestep'] = int(float(model.args['train_on_data_until_timestep']))
                predicted_data.loc[(participant, timestep+1), 'model_id'] = current_dir
        return predicted_data
            

    
    def predict_with_latest_models(self, experiment_dir: str):
        predicted_data = pd.DataFrame(index=self.data.index, columns=self.data.columns)
        predicted_data[['model_timestep', 'model_id']] = np.nan
        print('Making predictions...')
        model_catalogue = eval_reallabor_utils.ModelCatalogue(experiment_dir)
        for participant in tqdm(self.participants):
            participant_data = self.data.loc[participant]
            latest_dir = model_catalogue.get_current_model_dir(participant, participant_data.index[-1])
            if latest_dir is None:
                continue
            model, _ = eval_reallabor_utils.load_model_and_data(latest_dir)
            for timestep in participant_data.index[5:-1]:                    
                emas = tc.tensor(participant_data.loc[timestep-4:timestep, ema_names].values).float() 
                inputs = tc.tensor(participant_data.loc[timestep-4:timestep, intervention_names].values).float() 
                predicted_traj, _ = model.generate_free_trajectory(emas[-1:], inputs[-1:], 2,
                                                                    prewarm_data=emas[:-1], 
                                                                    prewarm_inputs=inputs[:-1]) 
                predicted_data.loc[(participant, timestep+1), ema_names] = predicted_traj[1].numpy()  
                predicted_data.loc[(participant, timestep+1), 'model_timestep'] = int(float(model.args['train_on_data_until_timestep']))
                predicted_data.loc[(participant, timestep+1), 'model_id'] = latest_dir         
        return predicted_data
    

    def aggregate_effects(self, participant: int, from_timestep: int=None, until_timestep: int=None,
                          rank: bool=False, sum_over_emas: bool=True, use_predicted: int=False):
        effects_df = self.calc_effects(use_predicted, EMI_only=True)
        sum_effects = effects_df.unstack('participant').xs(participant, axis=1, level=1)
        sum_effects = sum_effects.dropna()
        if from_timestep is not None:
            sum_effects = sum_effects.loc[from_timestep:]
        if until_timestep is not None:
            sum_effects = sum_effects[:until_timestep]
        EMI_occurrence = sum_effects[intervention_names].sum()
        # Get the sum of effects of each intervention on each EMA
        sum_effects = sum_effects[ema_names].T.dot(sum_effects[intervention_names])
        # Normalize them by the number of intervention presentations
        sum_effects = sum_effects.div(EMI_occurrence)
        if sum_over_emas:
            sum_effects = sum_effects.sum()
        if rank:
            sum_effects = sum_effects.rank(method='first').astype(int) - 1
        return sum_effects
    

if __name__=='__main__':
    raw_data_dir = data_utils.join_base_path('reallaborai4u/data_management/unprocessed_csv')
    model_dir = data_utils.join_base_path('bptt/results/MRT1_EveryDay02_Smoothed_best_runs')
    E = EffectsPrediction(raw_data_dir, model_dir, model_for_prediction='current')
    E.aggregate_effects(12)