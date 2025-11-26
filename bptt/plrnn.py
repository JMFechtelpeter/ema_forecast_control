import os
from operator import itemgetter
from typing import Optional
from argparse import Namespace
import torch as tc
import torch.nn as nn
import math
from dataset.multimodal_dataset import MultimodalDataset
import utils

class PLRNN(nn.Module):

    LATENT_MODELS = ['shallow-PLRNN', 'clipped-shallow-PLRNN', 'ALRNN']

    def __init__(self, args: Optional[Namespace|dict]=None, dataset: Optional[MultimodalDataset]=None, 
                 load_model_path: Optional[str]=None, resume_epoch: Optional[int]=None):

        super().__init__()
        self.device = None
        
        if args is not None:
            if not isinstance(args, dict):
                args = vars(args)

            if args['load_model_path'] is not None:
                self.init_from_model_path(args['load_model_path'], resume_epoch=args['resume_epoch'])
            else:
                self.args = args
                self.init_shapes()
                self.init_parameters()
                self.init_preprocessing()
                if dataset is not None:
                    nanmean = dataset.reference_data.nanmean(axis=0, keepdims=True).nan_to_num(nan=0)
                    self.set_data_mean(nanmean)

        elif load_model_path is not None:
            self.init_from_model_path(load_model_path, resume_epoch)

    # for backwards compatibility
    def init_from_model_path(self, load_model_path: str, resume_epoch: Optional[int]=None, backwards_compatibility: bool=True):
        if resume_epoch is None:
            resume_epoch = utils.infer_latest_epoch(load_model_path)
        self.args = utils.load_args(load_model_path)
        self.init_shapes()
        self.init_parameters()
        self.init_preprocessing()
        path = os.path.join(load_model_path, f'model_{resume_epoch}.pt')
        state_dict = tc.load(path)
        if backwards_compatibility:
            for old_key, new_key in self.old_state_dict_map.items():
                if old_key in state_dict.keys():
                    state_dict[new_key] = state_dict.pop(old_key)
        self.load_state_dict(state_dict)        

    # def to(self, device: str):
    #     self.device = device
    #     self.data_mean = self.data_mean.to(device)
    #     for name, param in self.parameters_.items():
    #         if param is not None:
    #             self.parameters_[name] = param.to(device)
    #     for name, param in self.named_parameters():
    #         if param is not None:
    #             self.register_parameter(name, param.to(device))

    def set_data_mean(self, data_mean: tc.Tensor):
        self.data_mean.data = data_mean
    
    def forward(self, X: tc.Tensor, inputs: Optional[tc.Tensor]=None, z0: Optional[tc.Tensor]=None,
                tf_alpha: float=0.125, initial_alpha: float=1.0, return_hidden=False, **kwargs):
        '''
            X and inputs have shape (batch x time x features)
        '''
        params = self.get_parameters()
        # if self.args['boxcox']:
        #     X = self.boxcox_step(X)
        if self.args['mean_centering']:
            X = X - self.data_mean.unsqueeze(1)
        # if self.args['learn_z0']:
        #     z0 = self.z0_model(X[0])
        if self.args['dim_x_proj'] > 0:
            B = params['B'] 
            X = self.observation_model_inverse_step(X, B)
        else:
            B = None

        if z0 is None:
            z0 = tc.zeros((X.shape[0], self.args['dim_z']), device=X.device)
            z0 = self.teacher_force(z0, X[:, 0], alpha=1.0)
        else:
            z0 = self.teacher_force(z0, X[:, 0], alpha=initial_alpha)
        
        # A, W1, W2, h1, h2, C = itemgetter('A', 'W1', 'W2', 'h1', 'h2', 'C')(params)
        Z = self.latent_sequence(z0, inputs=inputs, forcing_signal=X, tf_alpha=tf_alpha)
        
        if self.args['dim_x_proj'] > 0:
            output = self.observation_model_step(Z[:,:,:self.args['dim_x_proj']], B)
        else:
            output = Z[:,:,:self.args['dim_x']]
        if self.args['mean_centering']:
            output = output + self.data_mean.unsqueeze(1)
        # if self.args['boxcox']:
        #     output = self.boxcox_inverse_step(output)
        if return_hidden:
            return output, Z
        else:
            return output

    @tc.no_grad()
    def generate_free_trajectory(self, x0: tc.Tensor, T: int, inputs: Optional[tc.Tensor]=None, z0: Optional[tc.Tensor]=None,
                                 initial_alpha: float=1.0,
                                 prewarm_data: Optional[tc.Tensor]=None, prewarm_inputs: Optional[tc.Tensor]=None, prewarm_alpha: float=0.125,
                                 return_hidden: bool=False, recognition_matrix: Optional[tc.Tensor]=None, observation_matrix: Optional[tc.Tensor]=None):
        '''
            Given an initial x0 and inputs u0 ... u(T-1), predict x1 ... xT freely without teacher forcing.
            x0 has shape (feature)
            inputs, prewarm_data and prewarm_inputs have shape (time x features)
            Make sure that x0 and prewarm_data are disjoint, i.e. x0 is not the last value from prewarm_data!
        '''
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

        params = self.get_parameters()
        # if self.args['boxcox']:
        #     x0 = self.boxcox_step(x0)
        if self.args['mean_centering']:
            x0 = x0 - self.data_mean
        # if self.args['learn_z0']:
        #     z0 = self.z0_model(x0)
        if self.args['dim_x_proj'] > 0:
            if observation_matrix is not None:
                B = observation_matrix
            else:
                B = params['B']
            if recognition_matrix is not None:
                x0 = self.recognition_model_step(x0, recognition_matrix)
            else:
                x0 = self.observation_model_inverse_step(x0, B)
        else:
            B = None
        # if not self.args['learn_z0']:

        if prewarm_data is not None:
            _, Z = self.forward(prewarm_data, inputs=prewarm_inputs, z0=z0, tf_alpha=prewarm_alpha, return_hidden=True)
            z0 = Z[:, -1]
            z0 = self.teacher_force(z0, x0, alpha=prewarm_alpha)
        else:
            if z0 is None:
                z0 = tc.zeros((x0.shape[0], self.args['dim_z']), device=x0.device)
                z0 = self.teacher_force(z0, x0, alpha=1.0)
            else:
                z0 = self.teacher_force(z0, x0, alpha=initial_alpha)

        # A, W1, W2, h1, h2, C = itemgetter('A', 'W1', 'W2', 'h1', 'h2', 'C')(params)
        Z = self.latent_sequence(z0, inputs=inputs, T=T)
        
        if self.args['dim_x_proj'] > 0:
            output = self.observation_model_step(Z[:,:,:self.args['dim_x_proj']], B)
        else:
            output = Z[:,:,:self.args['dim_x']]
        if self.args['mean_centering']:
            output = output + self.data_mean.unsqueeze(1)
        # if self.args['boxcox']:
        #     output = self.boxcox_inverse_step(output)
        output = output.squeeze(0)
        Z = Z.squeeze(0)
        if return_hidden:
            return output, Z
        else:
            return output
        
    def jacobian(self, z: tc.Tensor):
        A, W1, W2, h2 = itemgetter('A','W1','W2','h2')(self.get_parameters())
        if self.args['latent_model']=='shallow-PLRNN':
            relu_indicator = (tc.einsum('ij,bj->bi', W2, z) + h2 > 0).float()
        elif self.args['latent_model']=='clipped-shallow-PLRNN':
            relu_indicator = (tc.einsum('ij,bj->bi', W2, z) + h2 > 0).float() - (tc.einsum('ij,bj->bi', W2, z) > 0).float()
        else:
            raise ValueError(f'Jacobian for latent model {self.args["latent_model"]} not implemented')
        jac = tc.diag(A) + tc.einsum('bij,jk->bik', tc.einsum('ij,bj->bij', W1, relu_indicator), W2)
        return jac.detach()

    # @classmethod
    def latent_sequence(self, z0, inputs=None, forcing_signal=None, tf_alpha=None, T=None):
    # def latent_sequence(cls, latent_model:str, z0, A, W1, W2, h1, h2, C=None, inputs=None, forcing_signal=None, tf_alpha=0.125, T=None, p=None):
        '''
        forcing signal and inputs have shape (time x batch x feature)
        z0 has shape (batch x feature)
        '''
        latent_step, latent_params, latent_kwargs = self.prepare_latent_step()

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
            z = latent_step(z, inputs[t], *latent_params, **latent_kwargs)
            Z[t] = z

        Z = Z.permute(1,0,2)

        return Z
    
    def latent_step(self, z, s=None):
        latent_step, latent_params, latent_kwargs = self.prepare_latent_step()
        return latent_step(z, s, *latent_params, **latent_kwargs)

    @classmethod
    def shallow_PLRNN_step(cls, z, s, A, W1, W2, h1, h2, C):
        z_ = A * z + tc.einsum('ij,bj->bi', W1, tc.relu(tc.einsum('ij,bj->bi', W2, z) + h2)) + h1
        if C is not None and s is not None:
            z_ += tc.einsum('ij,bj->bi', C, s)
        return z_
    
    @classmethod
    def clipped_shallow_PLRNN_step(cls, z, s, A, W1, W2, h1, h2, C):
        z_ = A * z + tc.einsum('ij,bj->bi', W1, tc.relu(tc.einsum('ij,bj->bi', W2, z) + h2) - tc.relu(tc.einsum('ij,bj->bi', W2, z))) + h1
        if C is not None and s is not None:
            z_ += tc.einsum('ij,bj->bi', C, s)
        return z_
    
    @classmethod
    def ALRNN_step(cls, z, s, A, W, h, C, p=1):
        z_partial_relu = tc.clone(z)
        z_partial_relu[:, -p:] = tc.relu(z_partial_relu[:, -p:])
        z_ = A * z + tc.einsum('ij,bj->bi', W, z_partial_relu) + h
        if C is not None and s is not None:
            z_ += tc.einsum('ij,bj->bi', C, s)
        return z_
    
    def prepare_latent_step(self):
        if self.args['latent_model'] == 'ALRNN':
            latent_params = itemgetter('A', 'W', 'h', 'C')(self.get_parameters())
            latent_step = self.ALRNN_step
            latent_kwargs = {'p': self.args['nonlinear_units']}
        elif self.args['latent_model'] == 'shallow-PLRNN':
            latent_params = itemgetter('A', 'W1', 'W2', 'h1', 'h2', 'C')(self.get_parameters())
            latent_step = self.shallow_PLRNN_step
            latent_kwargs = {}
        elif self.args['latent_model'] == 'clipped-shallow-PLRNN':
            latent_params = itemgetter('A', 'W1', 'W2', 'h1', 'h2', 'C')(self.get_parameters())
            latent_step = self.clipped_shallow_PLRNN_step
            latent_kwargs = {}
        else:
            raise NotImplementedError(f'Latent model {self.args["latent_model"]} not implemented')
        return latent_step, latent_params, latent_kwargs
    
    
    @staticmethod
    def teacher_force(z, forcing_signal, alpha=0.125):
        '''
        z and forcing_signal have shape (batch x feature)
        '''
        valid_map = ~forcing_signal.isnan() # creates the mask indicating which elements in x is not NaN
        z[:, :forcing_signal.shape[1]][valid_map] = (alpha * forcing_signal
                                                        + (1-alpha) * z[:, :forcing_signal.shape[1]])[valid_map]
        return z
    
    def infer_z0(self, x0: tc.Tensor):
        '''
        infer z0 by teacher forcing with alpha=1 from x0
        x0 has shape (batch x features)
        '''
        z0 = tc.zeros((x0.shape[0], self.args['dim_z']), device=x0.device) 
        if self.args['dim_x_proj'] > 0:
            B = self.get_parameters()['B']         
            forcing_signal = self.observation_model_inverse_step(x0, B)
        else:
            forcing_signal = x0
        z0 = self.teacher_force(z0, forcing_signal, alpha=1)
        return z0
    
    def observation_model_step(self, xproj, B):
        ''' xproj has shape (batch x time x features) or (batch x features)'''
        valid = ~xproj.isnan().any(axis=-1)
        res = tc.zeros(*xproj.shape[:-1], self.args['dim_x'], device=xproj.device) * tc.nan
        # res[valid] = tc.einsum('bxp,...bp->...bx', B, xproj[valid])
        res[valid] = tc.einsum('xp,...p->...x', B, xproj[valid])
        return res
    
    def observation_model_inverse_step(self, x, B):
        ''' x has shape (batch x time x features) or (batch x features)'''
        inv = tc.pinverse(B)
        valid = ~x.isnan().any(axis=-1)
        res = tc.zeros(*x.shape[:-1], self.args['dim_x_proj'], device=x.device) * tc.nan
        # res[valid] = tc.einsum('bpx,...bx->...bp', inv, x[valid])
        res[valid] = tc.einsum('px,...x->...p', inv, x[valid])
        return res

    def recognition_model_step(self, x, recognition_matrix):
        ''' x has shape (batch x time x features) or (batch x features)'''
        valid = ~x.isnan().any(axis=-1)
        res = tc.zeros(*x.shape[:-1], self.args['dim_z'], device=x.device) * tc.nan
        # res[valid] = tc.einsum('bzx,...bx->...bz', recognition_matrix, x[valid])
        res[valid] = tc.einsum('zx,...x->...z', recognition_matrix, x[valid])
        return res
    
    def get_parameters(self):
        return self.parameters_
    
    def init_shapes(self):
        self.shapes = dict()
        self.shapes['A'] = (self.args['dim_z'],)
        if self.args['latent_model'] == 'ALRNN':
            self.shapes['W'] = (self.args['dim_z'], self.args['dim_z'])
            self.shapes['h'] = (self.args['dim_z'],)
        elif self.args['latent_model'] in ['shallow-PLRNN', 'clipped-shallow-PLRNN']:
            self.shapes['W1'] = (self.args['dim_z'], self.args['dim_y'])
            self.shapes['W2'] = (self.args['dim_y'], self.args['dim_z'])
            self.shapes['h1'] = (self.args['dim_z'],)
            self.shapes['h2'] = (self.args['dim_y'],)
        if self.args['dim_s'] is not None:
            self.shapes['C'] = (self.args['dim_z'], self.args['dim_s'])
        else:
            self.shapes['C'] = None
        if self.args['dim_x_proj'] > 0:
            self.shapes['B'] = (self.args['dim_x'], self.args['dim_x_proj'])
        else:
            self.shapes['B'] = None

    def init_parameters(self):
        self.parameters_ = nn.ParameterDict()
        for name, shape in self.shapes.items():
            if shape is not None:
                self.parameters_[name] = self.init_uniform(shape)
            else:
                self.parameters_[name] = None

    def init_preprocessing(self):
        self.data_mean = nn.Parameter(tc.zeros(1, self.args['dim_x']), requires_grad=False)

    def init_xavier_uniform(self, shape, gain=.1):
        tensor = tc.empty(shape)
        nn.init.xavier_uniform_(tensor, gain=gain)
        return nn.Parameter(tensor, requires_grad=True)
    
    def init_uniform(self, shape, gain=1):
        tensor = tc.empty(shape)
        r = 1 / math.sqrt(shape[-1]) * gain
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)

    # def batch_unsqueeze(*tensors):
    #     tensors = list(tensors)
    #     for i in range(len(tensors)):
    #         tensors[i] = tensors[i].unsqueeze(0)
    #     return tuple(tensors)
 
    old_state_dict_map = {'latent_model.latent_step.A':'parameters_.A',
                        'latent_model.latent_step.W1':'parameters_.W1',
                        'latent_model.latent_step.W2':'parameters_.W2',
                        'latent_model.latent_step.h1':'parameters_.h1',
                        'latent_model.latent_step.h2':'parameters_.h2',
                        'latent_model.latent_step.C':'parameters_.C',
                        'obs_model.layer.weight':'parameters_.B',
                        'mean_center_layer.mean':'data_mean'}