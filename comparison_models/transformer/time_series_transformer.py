import os
from argparse import Namespace
from typing import Optional
import torch as tc
import torch.nn as nn
import math

import utils

class AutoregressiveTransformer(nn.Module):

    def __init__(self, args: Optional[dict|Namespace]=None,
                 load_model_path: Optional[str]=None, resume_epoch: Optional[int]=None):
        super().__init__()
        self.model = None
        self.device = None

        if args is not None:
            if not isinstance(args, dict):
                args = vars(args)
            self.args = utils.complement_args(args)

            if args['load_model_path'] is not None:
                self.init_from_model_path(args['load_model_path'], resume_epoch=args['resume_epoch'])
            else:
                self.args = args
                self.init_submodules()
        elif load_model_path is not None:
            self.init_from_model_path(load_model_path, resume_epoch)

    def init_from_model_path(self, load_model_path: str, resume_epoch: Optional[int]=None):
        # load argumentsn_steps_ahead_pred_mse
        self.args = utils.load_args(load_model_path)
        # init using arguments
        self.init_submodules()
        # restore model parameters
        path = os.path.join(load_model_path, '{}_{}.pt'.format('model', str(resume_epoch)))
        state_dict = tc.load(path)
        self.load_state_dict(state_dict)
    
    def to_device(self, device):
        self.device = device
        self.to(device)
    
    def init_submodules(self):
        args = self.args

        self.combined_input_size = args['dim_x'] + args['dim_s']

        self.encoder_input_layer = nn.Linear(self.combined_input_size, args['dim_model'])
        self.positional_encoding_layer = PositionalEncoder(d_model=args['dim_model'], dropout=self.args['dropout'], 
                                                           max_seq_len=args['max_seq_len'])
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=args['dim_model'], nhead=args['n_heads'], 
                                                   dim_feedforward=args['dim_feedforward'], dropout=args['dropout'],
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args['n_encoder_layers'], enable_nested_tensor=False)

        self.decoder_input_layer = nn.Linear(self.combined_input_size, args['dim_model'])
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=args['dim_model'], nhead=args['n_heads'], 
                                                   dim_feedforward=args['dim_feedforward'], dropout=args['dropout'],
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args['n_decoder_layers'])

        self.linear_mapping = nn.Linear(args['dim_model'], args['dim_x'])

    def forward(self, data: tc.Tensor, inputs: Optional[tc.Tensor], mask_past: bool=True, **kwargs):
        if inputs is not None:
            combined_input = tc.cat([data, inputs], dim=-1)
        else:
            combined_input = data
        dsl = self.args['decoder_seq_len']
        t = data.shape[1]
        src = combined_input[:, :-dsl]
        tgt = combined_input[:, -dsl-1:-1]
        if mask_past:
            src_mask = self.get_decoder_mask(dsl, t-dsl)
            tgt_mask = self.get_decoder_mask(dsl, dsl)
        else:
            src_mask, tgt_mask = None, None
        return self.forward_transformer(src, tgt, src_mask, tgt_mask)


    def forward_transformer(self, src: tc.Tensor, tgt: tc.Tensor, src_mask: Optional[tc.Tensor]=None, tgt_mask: Optional[tc.Tensor]=None):
        # if self.mean_center_layer is not None:
        #     src = self.mean_center_layer(src)
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src)
        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
        decoder_output = self.linear_mapping(decoder_output)
        # if self.mean_center_layer is not None:
        #     decoder_output = self.mean_center_layer.inverse(decoder_output)

        return decoder_output
    
    def get_decoder_mask(self, dim1: int, dim2: int, device=None):
        if device is None:
            device = self.device    
        return tc.triu(tc.ones(dim1, dim2) * float('-inf'), diagonal=1).to(self.device)
    
    def generate_free_trajectory(self, x0: tc.Tensor, steps: int, inputs: Optional[tc.Tensor]=None, 
                                 prewarm_data: Optional[tc.Tensor]=None, prewarm_inputs: Optional[tc.Tensor]=None, 
                                 *args, **kwargs):

        if x0.ndim>1:
            raise ValueError('x0 should have shape [dim_x]')
        x0 = x0.unsqueeze(0).unsqueeze(0)
        if prewarm_data is not None and prewarm_data.ndim==2:
            prewarm_data = prewarm_data.unsqueeze(0)
        if inputs is not None and inputs.ndim==2:
            inputs = inputs.unsqueeze(0)
        if prewarm_inputs is not None and prewarm_inputs.ndim==2:
            prewarm_inputs = prewarm_inputs.unsqueeze(0)
        with tc.no_grad():
            if prewarm_data is not None:
                if prewarm_inputs is not None and inputs is not None:
                    src = tc.cat([tc.cat([prewarm_data, prewarm_inputs], dim=-1),
                                tc.cat([x0, inputs[:, :1]], dim=-1)],
                                dim=1)
                else:
                    src = tc.cat([prewarm_data, x0], dim=1)
            else:
                if inputs is not None:
                    src = tc.cat([x0, inputs[:, :1]], dim=-1)
                else:
                    src = x0
            tgt_data = x0
            if inputs is not None:
                tgt = tc.cat([tgt_data, inputs[:, :1]], dim=-1)
            else:
                tgt = tgt_data
            # Iteratively concatenate tgt with the first element in the prediction
            for t in range(steps):
                tgt_mask = self.get_decoder_mask(tgt.shape[1], tgt.shape[1])
                src_mask = self.get_decoder_mask(tgt.shape[1], src.shape[1])
                pred = self.forward_transformer(src, tgt, src_mask, tgt_mask) 
                tgt_data = tc.cat([tgt_data, pred[:, -1:]], dim=1)      # Append new prediction step to tgt_data
                if t < steps-1:
                    if inputs is not None:
                        tgt = tc.cat([tgt_data, inputs[:, :t+2]], dim=-1)      # Combine tgt_data and one more step from inputs to new tgt
                    else:
                        tgt = tgt_data
        
        tgt_data = tgt_data.squeeze(0)

        return tgt_data[1:], 'placeholder_for_latent_traj_for_compatibility'
    
    def plot_prediction(self, *args, **kwargs):
        raise NotImplementedError('TODO: implement plot_prediction')
    
    def plot_generated_against_obs(self, *args, **kwargs):
        raise NotImplementedError('TODO: implement plot_generated_against_obs')
    
    def count_parameters(self):
        return sum([p.numel() for p in self.parameters()])
            

class PositionalEncoder(nn.Module):

    def __init__(self, dropout: float=0.1, max_seq_len: int=5000, d_model: int=512, batch_first: bool=False):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.d_model = d_model        
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = tc.arange(max_seq_len).unsqueeze(1)        
        div_term = tc.exp(tc.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))        
        pe = tc.zeros(max_seq_len, 1, d_model)        
        pe[:, 0, 0::2] = tc.sin(position * div_term)        
        pe[:, 0, 1::2] = tc.cos(position * div_term)        
        self.register_buffer('pe', pe)
        
    def forward(self, x: tc.Tensor) -> tc.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)
    

# class MeanCenterLayer(nn.Module):
#     def __init__(self, size: int):
#         super().__init__()
#         self.mean = nn.Parameter(tc.zeros((1, size)), requires_grad=False)
    
#     def update(self, reference_data: tc.Tensor):
#         if reference_data is not None:
#             nanmean = reference_data.nanmean(axis=0, keepdims=True)
#             nanmean = tc.nan_to_num(nanmean, nan=0)
#             self.mean.copy_(nanmean)

#     def forward(self, x):
#         if x.dim()==3:
#             return x - self.mean.unsqueeze(0)
#         else:
#             return x - self.mean
    
#     def inverse(self, x):
#         if x.dim()==3:
#             return x + self.mean.unsqueeze(0)
#         else:
#             return x + self.mean