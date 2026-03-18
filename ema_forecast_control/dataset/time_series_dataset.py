from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch as tc
from typing import Optional

import logging
log = logging.getLogger(__name__)



class TimeSeriesDataset(Dataset):
    def __init__(self, data: tc.Tensor, inputs: tc.Tensor,                 
                 name: str='', seq_len: int=0, batch_size: Optional[int]=None, bpe: Optional[int]=None, 
                 partial_missings_are_valid: bool=False,
                 tolerate_reduced_seq_len: bool=True,
                 max_valid_data_ratio: Optional[float]=None,
                 verbose: str='print'):

        super().__init__()   
        self.name = name if name else f'dataset_{id(self)}' 
        self.seq_len = seq_len 
        self.partial_missings_are_valid = partial_missings_are_valid
        self.tolerate_reduced_seq_len = tolerate_reduced_seq_len
        self.verbose = verbose
        self.dropout_to = max_valid_data_ratio
        self._process_data(data, inputs)        
                              
    def _process_data(self, data: tc.Tensor, inputs: Optional[tc.Tensor]):
        """
        Set the valid (non-missing) indices, the valid indices for sequences,
        the removal of leading nans and the total length according to the reference time series data. 
        This method is called as soon as the first time series is appended to the dataset.
        """
        if self.partial_missings_are_valid:
            self.valid_indices = tc.arange(data.shape[0])[(~data.isnan()).any(axis=1)]
        else:
            self.valid_indices = tc.arange(data.shape[0])[(~data.isnan()).all(axis=1)]

        if len(self.valid_indices) == 0:
            log.error(f'Dataset {self.name} contains no valid data points.')
            raise ValueError(f'Dataset {self.name} contains no valid data points.')
        
        self.data = data[self.valid_indices[0]:]
        if inputs is not None:
            self.inputs = inputs[self.valid_indices[0]:]
            self.has_external_inputs = True
        else:
            self.inputs = None
            self.has_external_inputs = False
        self.valid_indices = self.valid_indices - self.valid_indices[0]
        self.n_valid = len(self.valid_indices)
        self.T = self.data.shape[0]

        if self.dropout_to is not None:
            valid_ratio = self.n_valid / self.T
            if valid_ratio > self.dropout_to:
                drop = int(self.n_valid - (self.dropout_to * self.T))
                if drop > 0:
                    # cannot drop the first valid index
                    drop_idx = self.valid_indices[tc.randperm(len(self.valid_indices)-1)[:drop] + 1]
                    data[drop_idx] = np.nan
        
        #Sequence Length == 0 means do not split the time series into sequences
        if self.seq_len==0:
            self.seq_len = self.T
        elif self.seq_len > self.T:
            if self.tolerate_reduced_seq_len:
                self.seq_len = self.T
                log.warning(f'Sequence length {self.seq_len} too long, dataset has length {self.T}. Sequence length will be set to this value.')
            else:
                log.error(f'Sequence length {self.seq_len} too long, dataset has length {self.T}.')
                raise ValueError(f'Sequence length {self.seq_len} too long, dataset has length {self.T}.')
            
        #Valid sequence indices are non-missing data points that are early enough
        #to draw a sequence of length seq_len starting from there
        self.valid_sequence_indices = self.valid_indices[self.valid_indices <= self.T - self.seq_len]
    
    def __repr__(self):
        return (f'Dataset {self.name}, length {self.T}')

    def __len__(self):
        return len(self.valid_sequence_indices)

    def __getitem__(self, idx):
        valid_idx = self.valid_sequence_indices[idx]
        if self.inputs is not None:
            return self.data[valid_idx : valid_idx + self.seq_len], self.inputs[valid_idx : valid_idx + self.seq_len]
        else:
            return self.data[valid_idx : valid_idx + self.seq_len], None
    
    def get_dataloader(self, batch_size, shuffle=True, drop_last=True, **kwargs):
        """
        Returns a pytorch dataloader with sequences
        """        
        def list_collate(batch):
            batch_data, batch_inputs = [], []
            for data_item in batch:
                batch_data.append(data_item[0])
                if self.has_external_inputs:
                    batch_inputs.append(data_item[1])
            batch_data = tc.stack(batch_data)
            if self.has_external_inputs:
                batch_inputs = tc.stack(batch_inputs)
            else:
                batch_inputs = None
            return batch_data, batch_inputs
        
        if batch_size > len(self.valid_sequence_indices):
            batch_size = len(self.valid_sequence_indices)
            message = (f'Warning: Batch size {batch_size} too large, time series contains only {len(self)} distinct '
                  f'sequences of length {self.seq_len}, since they must start at a non-missing data point. '
                  f'The resulting batch size is {batch_size}.')
            log.warning(message)

        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=list_collate, drop_last=drop_last, **kwargs)
        return dataloader
    
    def get_rand_batch_indices(self, batch_size):
        """
        Returns n indices of sequences in random order, where n is batch_size 
        """
        indices = np.random.permutation(len(self))[:batch_size]
        return indices

    def to(self, device: tc.device):
        self.data = self.data.to(device)
        if self.has_external_inputs:
            self.inputs = self.inputs.to(device)