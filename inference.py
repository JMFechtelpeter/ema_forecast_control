#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:26:25 2022

@author: janik
"""

import torch as tc
import matplotlib.pyplot as plt
from bptt.plrnn import Model
import feature_names as feat_names

class Decay:    
    @classmethod
    def linear(cls, score):
        decay_weights = tc.linspace(1,0,len(score)+1)
        return score * decay_weights[:-1]
    @classmethod
    def none(cls, score):
        return score


def model_informed_suggestion(model: Model, n_steps: int, decay: str='none', 
                              consolidation_task: bool=False, plot_path: str=None):
    
    n_ints = len(feat_names.INTERVENTION_NAMES)
    rest_inputs = model.args['dim_s'] - n_ints
    
    def get_score_weights():
        score_weights = []
        for feat in model.args['data_features']:
            score_weights.append(feat_names.SCORE_WEIGHTS[feat])
        return score_weights
    
    data = model.dataset.timeseries['emas'].data
    last_data_point = data[-1:]
    prewarm_data = data[-4:-1]
    prewarm_inputs = model.dataset.timeseries['inputs'].data[-4:-1]
    
    interventions = tc.vstack((tc.zeros((1, model.args['dim_s'])), 
                               tc.hstack((tc.eye(n_ints), 
                                          tc.zeros((n_ints, rest_inputs))))))
    scores = tc.zeros(len(feat_names.INTERVENTION_NAMES) + 1)
    for i, inter in enumerate(interventions):
        inputs = tc.vstack([inter.unsqueeze(0), tc.zeros((n_steps-1, model.args['dim_s']))])
        predicted_traj, _ = model.generate_free_trajectory(last_data_point, inputs, n_steps,
                                                           prewarm_data=prewarm_data, 
                                                           prewarm_inputs=prewarm_inputs)        
        decay_function = eval(f'Decay.{decay}')
        feat_weights = decay_function(tc.tensor(get_score_weights(), requires_grad=False))
        scores[i] = (predicted_traj * feat_weights.unsqueeze(0)).sum()
        
    if consolidation_task:
        distribution = tc.distributions.categorical.Categorical(logits=scores[1:])
        features = feat_names.CONSOLIDATION_NAMES
    else:
        distribution = tc.distributions.categorical.Categorical(logits=scores[1:])        
        features = feat_names.INTERVENTION_NAMES
        
    if plot_path is not None:
        plt.bar(range(len(features)), distribution.probs)
        plt.xticks(range(len(features)), features, rotation=45)
        plt.savefig(plot_path, dpi=300)
        # plt.close()        
    
    k = distribution.sample()
    return features[k]


def random_suggestion(model, consolidation_task=False):

    if consolidation_task:
        k = tc.randint(model.args['dim_s'])
        return feat_names.CONSOLIDATION_NAMES[k]
    else:
        k = tc.randint(model.args['dim_s'] + 1)
        return feat_names.INTERVENTION_NAMES[k]
    