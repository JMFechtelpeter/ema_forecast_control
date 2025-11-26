import os
from operator import itemgetter
from typing import Optional
from argparse import Namespace
import torch as tc
import torch.nn as nn
from matplotlib import pyplot as plt
import math
import utils
from dataset.multimodal_dataset import DatasetWrapper


class HierarchizedPLRNN(nn.Module):

    LATENT_MODELS = ['shallow-PLRNN', 'clipped-shallow-PLRNN']
    
    def __init__(self, args: Optional[Namespace|dict]=None, dataset_wrapper: Optional[DatasetWrapper]=None, 
                 load_model_path: Optional[str]=None, resume_epoch: Optional[int]=None, new_subjects: Optional[str]='add'):

        super().__init__()

        if load_model_path is not None:
            self.init_from_model_path(load_model_path, resume_epoch=resume_epoch)
            if dataset_wrapper is not None:
                if new_subjects=='add':
                    self.add_new_subjects(dataset_wrapper)
                elif new_subjects=='replace':
                    self.replace_subjects_with(dataset_wrapper)
        
        elif args is not None:
            if not isinstance(args, dict):
                args = vars(args)

            self.args = args
            self.init_shapes()
            self.init_shared()
            self.init_individual()
            self.init_preprocessing()
            self.init_subject_index_map()
            if dataset_wrapper is not None:
                nanmean = dataset_wrapper.get_nanmean().nan_to_num(nan=0)
                self.set_data_mean(nanmean)
        

    def init_from_model_path(self, load_model_path: str, resume_epoch: Optional[int]=None):
        if resume_epoch is None:
            resume_epoch = utils.infer_latest_epoch(load_model_path)
        self.args = utils.load_args(load_model_path)
        self.init_shapes()
        self.init_shared()
        self.init_individual()
        self.init_preprocessing()
        self.init_subject_index_map()
        path = os.path.join(load_model_path, '{}_{}.pt'.format('model', str(resume_epoch)))
        state_dict = tc.load(path)
        self.load_state_dict(state_dict)

    @property
    def n_subjects(self):
        return len(self.args['subject_indices'])
    
    def set_data_mean(self, data_mean: tc.Tensor):
        self.data_mean.data = data_mean
    
    def forward(self, subject_idx: tc.Tensor|int, X: tc.Tensor, inputs: Optional[tc.Tensor]=None, z0: Optional[tc.Tensor]=None,
                tf_alpha: float=0.125, return_hidden: bool=False, subject_idx_is_integer: bool=False):
        '''
            X and inputs have shape (batch x time x features)
        '''
        if subject_idx_is_integer:
            subject_int_idx = subject_idx
        else:
            subject_int_idx = self.get_subject_int_idx(subject_idx)        

        params = self.get_individual_params(subject_int_idx, subject_idx_is_integer=True)        
        # if self.args['boxcox']:
        #     X = self.boxcox_step(X)
        if self.args['mean_centering']:
            X = X - self.data_mean[subject_int_idx].unsqueeze(1)
        if self.args['dim_x_proj'] > 0:
            B = params['B']
            X = self.observation_model_inverse_step(X, B)
        else:
            B = None

        if z0 is None:
            z0 = tc.zeros((X.shape[0], self.args['dim_z']), device=X.device)
        z0 = self.teacher_force(z0, X[:, 0], alpha=1)               
        A, W1, W2, h1, h2, C = itemgetter('A', 'W1', 'W2', 'h1', 'h2', 'C')(params)
        Z = self.PLRNN_sequence(z0, A, W1, W2, h1, h2, C=C, inputs=inputs, forcing_signal=X, tf_alpha=tf_alpha)
        
        if self.args['dim_x_proj'] > 0:
            output = self.observation_model_step(Z[:,:,:self.args['dim_x_proj']], B)
        else:
            output = Z[:,:,:self.args['dim_x']]
        if self.args['mean_centering']:
            output = output + self.data_mean[subject_int_idx].unsqueeze(1)
        # if self.args['boxcox']:
        #     output = self.boxcox_inverse_step(output)
        
        if return_hidden:
            return output, Z
        else:
            return output

    @tc.no_grad()
    def generate_free_trajectory(self, subject_idx: tc.Tensor|int, x0: tc.Tensor, T: int, inputs: Optional[tc.Tensor]=None,
                                 z0: Optional[tc.Tensor]=None, 
                                 prewarm_data: Optional[tc.Tensor]=None, prewarm_inputs: Optional[tc.Tensor]=None, prewarm_alpha: float=0.125,
                                 return_hidden: bool=False, subject_idx_is_integer: bool=False) -> tuple|tc.Tensor:
        '''
            x0 has shape (batch x feature) or (feature)
            inputs, prewarm_data and prewarm_inputs have shape (batch x time x features) or (time x features)
        '''
        if isinstance(subject_idx, int) or subject_idx.ndim == 0:
            squeeze = True
            if x0.ndim == 1:
                x0 = x0.unsqueeze(0)
            if inputs is not None and inputs.ndim == 2:
                inputs = inputs.unsqueeze(0)
            if prewarm_data is not None and prewarm_data.ndim == 2:
                prewarm_data = prewarm_data.unsqueeze(0)
            if prewarm_inputs is not None and prewarm_inputs.ndim == 2:
                prewarm_inputs = prewarm_inputs.unsqueeze(0)
            if z0 is not None and z0.ndim == 1:
                z0 = z0.unsqueeze(0)
        else:
            squeeze = False
        if subject_idx_is_integer:
            subject_int_idx = subject_idx
        else:
            subject_int_idx = self.get_subject_int_idx(subject_idx)
            
        params = self.get_individual_params(subject_int_idx, subject_idx_is_integer=True)
        # if self.args['boxcox']:
        #     x0 = self.boxcox_step(x0)
        if self.args['mean_centering']:
            x0 = x0 - self.data_mean[subject_int_idx]
        # if self.args['learn_z0']:
        #     z0 = self.z0_model(x0)
        if self.args['dim_x_proj'] > 0:
            B = params['B']
            x0 = self.observation_model_inverse_step(x0, B)
        else:
            B = None
        # if not self.args['learn_z0']:

        if prewarm_data is not None:
            _, Z = self.forward(subject_int_idx, prewarm_data, inputs=prewarm_inputs, z0=z0, tf_alpha=prewarm_alpha, return_hidden=True,
                                subject_idx_is_integer=True)
            z0 = Z[:, -1]
            z0 = self.teacher_force(z0, x0, alpha=prewarm_alpha)
        else:
            if z0 is None:
                z0 = tc.zeros((x0.shape[0], self.args['dim_z']), device=x0.device)
            z0 = self.teacher_force(z0, x0, alpha=1)        

        A, W1, W2, h1, h2, C = itemgetter('A', 'W1', 'W2', 'h1', 'h2', 'C')(params)
        Z = self.PLRNN_sequence(z0, A, W1, W2, h1, h2, C=C, inputs=inputs, T=T)
        
        if self.args['dim_x_proj'] > 0:
            output = self.observation_model_step(Z[:,:,:self.args['dim_x_proj']], B)
        else:
            output = Z[:,:,:self.args['dim_x']]
        if self.args['mean_centering']:
            output = output + self.data_mean[subject_int_idx].unsqueeze(1)
        # if self.args['boxcox']:
        #     output = self.boxcox_inverse_step(output)

        if squeeze:
            output = output.squeeze(0)
            Z = Z.squeeze(0)
            
        if return_hidden:
            return output, Z
        else:
            return output

    def PLRNN_step(self, z, A, W1, W2, h1, h2, C=None, s=None):
        if self.args['latent_model'] == 'shallow-PLRNN':
            z_ = A * z + tc.einsum('bij,bj->bi', W1, tc.relu(tc.einsum('bij,bj->bi', W2, z) + h2)) + h1
        if self.args['latent_model'] == 'clipped-shallow-PLRNN':
            z_ = A * z + tc.einsum('bij,bj->bi', W1, tc.relu(tc.einsum('bij,bj->bi', W2, z) + h2) - tc.relu(tc.einsum('bij,bj->bi', W2, z))) + h1
        else:
            raise NotImplementedError(f'Latent model {self.args["latent_model"]} not implemented')
        if C is not None and s is not None:
            z_ += tc.einsum('bij,bj->bi', C, s)
        return z_
    
    def teacher_force(self, z, forcing_signal, alpha=0.125):
        '''
        z and forcing_signal have shape (batch x feature)
        '''
        valid_map = ~forcing_signal.isnan() # creates the mask indicating which elements in x is not NaN
        z[:, :forcing_signal.shape[1]][valid_map] = (alpha * forcing_signal
                                                        + (1-alpha) * z[:, :forcing_signal.shape[1]])[valid_map]
        return z
    
    def PLRNN_sequence(self, z0, A, W1, W2, h1, h2, C=None, inputs=None, 
                       forcing_signal=None, tf_alpha=0.125, T=None):
        '''
        forcing signal and inputs have shape (batch x time x feature)
        z0 has shape (batch x feature)
        '''
        if T is None:
            if forcing_signal is not None:
                T = forcing_signal.shape[1]
            else:
                raise ValueError('Sequence length T must be provided if forcing_signal is None.')
        if forcing_signal is not None:
            forcing_signal = forcing_signal.permute(1,0,2)
            T = min(T, forcing_signal.shape[0])
        if inputs is not None:
            inputs = inputs.permute(1,0,2)
            T = min(T, inputs.shape[0])
        else:
            inputs = [None] * T
        z = z0
        batch_size, n_feat = z0.shape
        Z = tc.empty(size=(T, batch_size, n_feat), device=z0.device)
        for t in range(T):
            if forcing_signal is not None:
                z = self.teacher_force(z, forcing_signal[t], alpha=tf_alpha)
            z = self.PLRNN_step(z, A, W1, W2, h1, h2, C=C, s=inputs[t])
            Z[t] = z

        Z = Z.permute(1,0,2)

        return Z
    
    def observation_model_step(self, xproj, B):
        ''' xproj has shape (batch x time x features) 
            B has shape (batch x dim_x x dim_x_proj) '''
        valid = ~xproj.isnan().any(axis=-1)
        res = tc.zeros(*xproj.shape[:-1], self.args['dim_x'], device=xproj.device) * tc.nan
        # res[valid] = tc.einsum('xp,...p->...x', B, xproj[valid])
        for b in range(B.shape[0]):
            res[b][valid[b]] = tc.einsum('xp,...p->...x', B[b], xproj[b][valid[b]])
        return res
    
    def observation_model_inverse_step(self, x, B):
        ''' x has shape (batch x time x features) or (batch x features)
            B has shape (batch x dim_x_proj x dim_x) '''
        inv = tc.pinverse(B)
        valid = ~x.isnan().any(axis=-1)
        res = tc.zeros(*x.shape[:-1], self.args['dim_x_proj'], device=x.device) * tc.nan
        # res[valid] = tc.einsum('bpx,...bx->...bp', inv, x[valid])
        for b in range(B.shape[0]):
            res[b][valid[b]] = tc.einsum('px,...x->...p', inv[b], x[b][valid[b]])
        return res
    
    def get_individual_params(self, subject_idx: Optional[int|tc.Tensor]=None, subject_idx_is_integer: bool=False):
        '''
            Returns model params in shape (subject x *param_shape)
        '''
        params = {}
        if subject_idx is None:
            subject_int_idx = tc.arange(self.n_subjects)
        elif subject_idx_is_integer:
            subject_int_idx = subject_idx
        else:
            subject_int_idx = self.get_subject_int_idx(subject_idx)
        for name, shape in self.shapes.items():
            rule = 'sp,pij->sij' if len(shape) > 1 else 'sp,pi->si'
            matrix = tc.einsum(rule, self.individual_parameters, self.shared_projection_matrices[name])
            params[name] = matrix[subject_int_idx]
        return params#, self.state_covariance[global_id]

    
    def init_shapes(self):
        self.shapes = dict()
        self.shapes['A'] = (self.args['dim_z'],)
        self.shapes['W1'] = (self.args['dim_z'], self.args['dim_y'])
        self.shapes['W2'] = (self.args['dim_y'], self.args['dim_z'])
        self.shapes['h1'] = (self.args['dim_z'],)
        self.shapes['h2'] = (self.args['dim_y'],)
        self.shapes['C'] = (self.args['dim_z'], self.args['dim_s'])
        if self.args['dim_x_proj'] > 0:
            self.shapes['B'] = (self.args['dim_x'], self.args['dim_x_proj'])

    def init_shared(self):
        self.shared_projection_matrices = nn.ParameterDict()
        for name, shape in self.shapes.items():
            dims = (self.args['dim_p'], *shape)
            self.shared_projection_matrices[name] = self.init_xavier_uniform(dims)

    def init_individual(self):
        self.individual_parameters = self.init_uniform(
            (self.n_subjects, self.args['dim_p'])
        )
        # self.state_covariance = self.init_constant(
        #     (self.n_subjects, self.args['dim_z'])
        # )

    def init_preprocessing(self):
        self.data_mean = nn.Parameter(tc.zeros((self.n_subjects, self.args['dim_x'])), requires_grad=False)

    def init_subject_index_map(self):
        self.subject_index_map = {self.args['subject_indices'][k]: k for k in range(len(self.args['subject_indices']))}

    def init_xavier_uniform(self, shape, gain=.1):
        tensor = tc.empty(*shape)
        nn.init.xavier_uniform_(tensor, gain=gain)
        return nn.Parameter(tensor, requires_grad=True)
    
    def init_uniform(self, shape, gain=1):
        tensor = tc.empty(*shape)
        r = 1 / math.sqrt(shape[-1]) * gain
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)
    
    def get_subject_int_idx(self, subject_idx: int|tc.Tensor) -> tc.Tensor:
        if isinstance(subject_idx, int):
            return tc.tensor([self.subject_index_map[subject_idx]])
        elif (isinstance(subject_idx, tc.Tensor) and subject_idx.ndim == 0):
            return tc.tensor([self.subject_index_map[subject_idx.item()]])
        else:
            return tc.tensor([self.subject_index_map[idx.item()] for idx in subject_idx])
        
    def freeze_shared(self):
        for param in self.shared_projection_matrices.values():
            param.requires_grad = False

    def unfreeze_shared(self):
        for param in self.shared_projection_matrices.values():
            param.requires_grad = True

    def add_new_subjects(self, dataset_wrapper: DatasetWrapper):
        subject_idx = dataset_wrapper.dataset_indices
        if any(idx in self.subject_index_map for idx in subject_idx):
            raise ValueError('Some of the new subjects are already present in the model.')
        for idx in subject_idx:
            if idx not in self.subject_index_map:
                self.args['subject_indices'].append(idx)
                self.subject_index_map[idx] = len(self.subject_index_map)
        self.individual_parameters = nn.Parameter(tc.cat([self.individual_parameters, self.init_uniform((len(subject_idx), self.args['dim_p']))], dim=0),
                                                  requires_grad=True)
        # self.state_covariance = tc.cat([self.state_covariance, self.init_constant((len(subject_idx), self.args['dim_z']))], dim=0)
        nanmean = tc.cat([self.data_mean.data, dataset_wrapper.get_nanmean().nan_to_num(nan=0)], dim=0)
        self.init_preprocessing()
        self.set_data_mean(nanmean)

    def replace_subjects_with(self, dataset_wrapper: DatasetWrapper):
        self.args['subject_indices'] = dataset_wrapper.dataset_indices
        self.init_individual()
        self.init_preprocessing()
        self.init_subject_index_map()
        nanmean = dataset_wrapper.get_nanmean().nan_to_num(nan=0)
        self.set_data_mean(nanmean)