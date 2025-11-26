# basic_dataset.py
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
import numpy as np
import pandas as pd
import torch as tc
from collections import OrderedDict, defaultdict
import itertools
import matplotlib.pyplot as plt
from typing import Optional
import attrs
import random
import logging
log = logging.getLogger(__name__)

@attrs.define
class TimeSeries():
    """
    Holds one time series and preprocessors for its train/test set respectively.
    Has attributes 'train_data' and 'test_data' which hold the preprocessed train/test set of the time series.
    """
    name: str
    data: tc.Tensor
    feature_names: list[str] = attrs.field()

    @feature_names.default
    def _default_feature_names(self):
        return [f'{self.name}[{k}]' for k in range(self.T)]

    @property
    def T(self): # returns  the length of the TS (number of time steps)
        return self.data.shape[0]
    
    @property 
    def dim(self): # returns the number of features 
        return self.data.shape[1]
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):# returns  the length of the TS (number of time steps) why 2 methods to return teh length of teh TS?
        return self.T
    
    def plot(self, axes=None, **kwargs):
        """
        Plot the preprocessed (if raw: raw) timeseries. If ax is none, create new figure.
        """
        newfig = (axes is None)
        if newfig:
            fig, axes = plt.subplots(self.dim, 1, figsize=(6.4, 1+1.6*self.dim), squeeze=False)       
            fig.suptitle(self.name)
        x = tc.arange(self.T)
        for i, ax in enumerate(axes):
            features = slice(i, i+1)
            ax.plot(x, self.data[:, features], **kwargs)
            if newfig:
                ax.title.set_text(self.feature_names[i])      
        return axes
    
    def to(self, device):
        self.data = self.data.to(device)


class MultimodalDataset(Dataset):
    """
    Universal Dataset subclass for ahead prediction, where the training is done on the first part
    and testing on the second part of a time series.
    seq_len:    Length of sequences drawn for training (if 0, it is maximized to the complete train length)
    ignore_leading_nans: Remove leading nans from the first timeseries, and
        remove the leading values of the other timeseries accordingly
    return_single_tensor: If true, calling the dataset, or calling dataset.test_item,
        will return a tensor containing all not-none timeseries
        concatenated along axis 1 (the feature axis). If false, it will return a list
        of tensors, one for each timeseries.
    tolerate_partial_missings: Defines how missing values in the reference timeseries are handled.
        If False, only data points with no missing features are considered valid (default)
        If True, data points with not all features missing are considered valid
    """
    def __init__(self, name: str='', seq_len: int=0, batch_size: Optional[int]=None, bpe: Optional[int]=None, 
                 ignore_leading_nans: bool=True, 
                 return_single_tensor: bool=False, 
                 tolerate_partial_missings: bool=False,
                 tolerate_reduced_seq_len: bool=True,
                 random_dropout_to_level: Optional[float]=None,
                 verbose: str='print'):

        super().__init__()   
        self.name = name if name else f'dataset_{id(self)}' 
        self.seq_len = seq_len 
        self.ignore_leading_nans = ignore_leading_nans
        self.return_tensor = return_single_tensor
        self.tolerate_partial_missings = tolerate_partial_missings
        self.tolerate_reduced_seq_len = tolerate_reduced_seq_len
        self.verbose = verbose
        self.timeseries = OrderedDict()
        self.dropout_to = random_dropout_to_level
        if batch_size is not None:
            log.warning('The global definition of a batch_size in a dataset is deprecated. The batch_size argument should be passed to the DataLoader instead.')
        if bpe is not None:
            log.warning('The bpe argument to a Dataset is deprecated. The number of batches per epoch should instead be handled by the training loop.')
        self.T = 0
                              
    def _reference(self, data):
        """
        Set the valid (non-missing) indices, the valid indices for sequences,
        the removal of leading nans and the total length according to the reference time series data. 
        This method is called as soon as the first time series is appended to the dataset.
        """
        #If leading nans are ignored, determine the number of leading data points to be deleted
        if self.tolerate_partial_missings:
            self.first_valid_index = pd.DataFrame(data).first_valid_index()
        else:
            self.first_valid_index = max([pd.Series(data[:,i]).first_valid_index() for i in range(data.shape[1])])
        if self.first_valid_index is None:
            raise ValueError(f'No valid data points in reference dataset {self.name}.')
        self.leading_nans = self.first_valid_index * 1
        if self.ignore_leading_nans:
            data = data[self.first_valid_index:]
            self.first_valid_index = 0
        if self.dropout_to is not None:
            valid_idx = tc.where((~data.isnan()).all(axis=1))[0]
            valid = len(valid_idx)
            valid_ratio = valid / (len(data))
            if valid_ratio > self.dropout_to:
                drop = int(valid - (self.dropout_to * len(data)))
                if drop > 0:
                    drop_idx = valid_idx[tc.randperm(len(valid_idx)-1)[:drop] + 1]
                    data[drop_idx] = np.nan

        # self.last_valid_index = pd.Series(data[:,0]).last_valid_index()
        #Set total number of time points
        self.T = data.shape[0]
        #Store reference data
        self.reference_data = data
        #Valid indices are all non-missing data indices.
        if self.tolerate_partial_missings:
            self.valid_indices = tc.arange(self.reference_data.shape[0])[(~self.reference_data.isnan()).any(axis=1)]
        else:
            self.valid_indices = tc.arange(self.reference_data.shape[0])[(~self.reference_data.isnan()).all(axis=1)]
        #Sequence Length == 0 means do not split the time series into sequences
        if self.seq_len==0:
            self.seq_len = self.T - self.first_valid_index
        elif self.seq_len > self.T - self.first_valid_index:
            message = (f'Sequence length {self.seq_len} too long. Maximum sequence length for this dataset is '
                                f'{self.T - self.first_valid_index}. Sequence length will be set to this value.')
            if self.tolerate_reduced_seq_len:
                self.seq_len = self.T - self.first_valid_index
                if self.verbose == 'print':
                    print(message)
                elif self.verbose == 'log':
                    log.warning(message)
            else:
                raise ValueError(message)
            
        #Valid sequence indices are non-missing data points that are early enough
        #to draw a sequence of length seq_len starting from there
        self.valid_sequence_indices = self.valid_indices[self.valid_indices <= self.T - self.seq_len]
    
    def __repr__(self):
        if len(self.timeseries)>0:
            return (f'Dataset of {self.T} time steps with timeseries'+'\n\t+'
                    +"\n\t+ ".join(self.timeseries.keys()) + '\n')
        else:
            return 'Empty Dataset'

    def __len__(self):
        return len(self.valid_sequence_indices)

    def __getitem__(self, idx):
        valid_idx = self.valid_sequence_indices[idx]
        x = []
        for key, sts in self.timeseries.items():
            if sts is not None:
                x.append(sts.data[valid_idx : valid_idx + self.seq_len])
            elif not self.return_tensor:
                x.append(None)
        if self.return_tensor:
            x = tc.cat(x, dim=1)
        return x
    
    def add_timeseries(self, data: Optional[np.ndarray|tc.Tensor], name: Optional[str]=None,
                       feature_names: list[str]=[]):
        """
        Add a TimeSeries object to the dataset.timeseries dictionary.
        data:   time series array of shape (time x features). If None, a None object
                will be added to dataset.timeseries instead of a TimeSeries.
        name:   key of the timeseries in dataset.timeseries      
        feature_names: collection of strings, indicating the names of the time series features.
                Useful for plotting.
        """
        if name is None:
            name = f'timeseries_{len(self.timeseries)+1}'
            
        if data is not None:
            if not isinstance(data, tc.Tensor):
                data = tc.tensor(data, dtype=tc.float32)
            else:
                data = data.clone().detach().float()
            if (data.shape[0] == 1) and len(data.shape)==3:
                data = data.squeeze(0)
            if len(self.timeseries)==0:
                self._reference(data)
                if self.ignore_leading_nans:
                    data = data[self.leading_nans:]            
            if self.T > data.shape[0]:
                missing = self.T - data.shape[0]
                data = tc.vstack([data, tc.zeros((missing, data.shape[1]))])
            else:
                data = data[:self.T]
            self.timeseries[name] = TimeSeries(name, data, feature_names)
            
        else:
            if len(self.timeseries)==0:
                raise ValueError('The first timeseries of a dataset cannot be None.')
            self.timeseries[name] = None

    def data(self, index=None):
        data = []
        for sts in self.timeseries.values():
            if sts is not None:
                if index is None:
                    data.append(sts.data)
                else:
                    data.append(sts.data[index])
            else:
                data.append(None)
        return tuple(data)
    
    def n_timesteps(self):
        return len(self.reference_data)
        
    def show_timeseries(self):
        """
        Prints info about the dataset.timeseries
        """
        for name, sts in self.timeseries.items():
            if sts is not None:
                print(f'{name}: {sts}')
            else:
                print(f'{name}: None')
    
    def get_dataloader(self, batch_size, shuffle=True, drop_last=True, **kwargs):
        """
        Returns a pytorch dataloader with sequences
        """        
        def list_collate(batch):
            batch_data = [[] for sts in self.timeseries]
            for data_item in batch:
                for i in range(len(self.timeseries)):
                    if data_item[i] is not None:
                        batch_data[i].append(data_item[i])
            for i in range(len(self.timeseries)):
                if len(batch_data[i]) > 0:
                    batch_data[i] = tc.stack(batch_data[i])
                else:
                    batch_data[i] = None
            return batch_data
        
        if batch_size > len(self.valid_sequence_indices):
            batch_size = len(self.valid_sequence_indices)
            message = (f'Warning: The reference time series contains only {len(self)} distinct '
                  f'sequences of length {self.seq_len}, since they must start at a non-missing data point. '
                  f'Consequently, the requested batch size {batch_size} is too large. '
                  f'The resulting batch size is {batch_size}.')
            log.warning(message)

        if self.return_tensor:
            dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
        else:
            dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=list_collate, drop_last=drop_last, **kwargs)
        return dataloader
    
    def get_rand_batch_indices(self, batch_size):
        """
        Returns n indices of sequences in random order, where n is batch_size 
        """
        indices = np.random.permutation(len(self))[:batch_size]
        return indices

    def to(self, device: tc.device):
        self.reference_data = self.reference_data.to(device)
        for sts in self.timeseries.values():
            if sts is not None:
                sts.to(device)


class DatasetWrapper(Dataset):
    """
    A dataset wrapper class that combines multiple `MultimodalDataset` instances and allows 
    unified operations on them. This class provides methods for accessing items across 
    multiple datasets, creating random data loaders, and managing valid sequences.

    Args:
        dataset_list (list[MultimodalDataset]): A list of `MultimodalDataset` instances to be wrapped.

    Attributes:
        dataset_list (list[MultimodalDataset]): List of datasets being managed by the wrapper.
        dataset_lengths (list[int]): List of lengths for each dataset, based on valid sequence indices.
        total_length (int): Sum of all valid sequence lengths across all datasets.
    """
    def __init__(self, datasets: dict[int,MultimodalDataset]|pd.Series):
        """
        Initializes the wrapper with a list of `MultimodalDataset` instances and computes 
        lengths for each dataset based on their valid sequences.

        Args:
            dataset_list (list[MultimodalDataset]): A list of datasets to be wrapped.
        """
        if isinstance(datasets, pd.Series):
            self.datasets = datasets
        else:
            self.datasets = pd.Series(datasets)
        self.check_timeseries_compatibility(self.datasets.to_list(), 'DatasetWrapper can only wrap datasets with the same timeseries.')
        self.timeseries_names = self.datasets.iloc[0].timeseries.keys()
        self.n_datasets = len(self.datasets)
        self.dataset_lengths = pd.Series([len(ds.valid_sequence_indices) for ds in self.datasets], index=self.datasets.index)
        self.cum_lengths = self.dataset_lengths.cumsum()
        self.total_length = sum(self.dataset_lengths)
        self.dataset_indices = self.datasets.index

    def __getitem__(self, idx: int|tuple):
        """
        Retrieves an item from the combined datasets based on a global index.

        Args:
            idx (int or tuple): The global index of the item to retrieve. If a tuple, it should 
                                be (global_idx, local_idx) where global_idx is the dataset index 
                                and local_idx is the index within that dataset.

        Returns:
            The data item corresponding to the given index.
        
        Raises:
            IndexError: If the index is out of range.
        """
        if isinstance(idx, int):
            global_integer_idx = np.searchsorted(self.cum_lengths, idx, side='right').item()
            local_idx = idx - self.cum_lengths.iloc[global_integer_idx]
            global_idx = self.datasets.index[global_integer_idx]
        elif isinstance(idx, tuple):
            global_idx, local_idx = idx
        else:
            raise IndexError("Index must be an integer or a tuple of (global_idx, local_idx)")
        dataset = self.datasets.loc[global_idx]
        if local_idx >= len(dataset.valid_sequence_indices):
            raise IndexError(f"Local index {local_idx} out of range for dataset {global_idx} with {len(dataset.valid_sequence_indices)} valid sequences")
        return global_idx, dataset[local_idx]
    
    def __len__(self):
        """
        Returns the total number of valid sequences across all datasets.

        Returns:
            int: Total length of all valid sequences.
        """
        return self.total_length

    def get_dataloader(self, subjects_per_batch: int, seq_per_subject: int, drop_last: bool=False, shuffle: bool=True, **kwargs):
        """
        Create a DataLoader for specific random sampling of sequences from the combined datasets.
        Uses a custom Sampler and BatchSampler to ensure the correct number of datasets per batch and the correct number of sequences per dataset.
        Args:
        - subjects_per_batch: Number of datasets to include in each batch
        - seq_per_subject: Number of sequences to include from each dataset in each batch
        """
        self.check_seq_len_compatibility(self.datasets.to_list(), 'DatasetWrapper can only create a DataLoader if sequence lengths are equal.')
        batch_sampler = HierarchicalBatchSampler(self, subjects_per_batch, seq_per_subject, drop_last, shuffle)
        return DataLoader(self, batch_sampler=batch_sampler, **kwargs)          
    
    def local_data(self, index: tuple[int, int|slice]) -> tuple:
        return self.datasets[index[0]].data(index[1])
    
    def global_data(self, index: int|slice, global_indices: Optional[list]=None) -> tuple:
        ts_data = defaultdict(list)
        lengths = []
        if global_indices is not None:
            items = self.datasets.loc[global_indices].items()
        else:
            items = self.datasets.items()
        for i, ds in items:
            subject_data = ds.data(index)
            lengths.append(len(subject_data[0]))
            for j, ts in enumerate(self.timeseries_names):
                if subject_data[j] is not None:
                    ts_data[ts].append(subject_data[j])
        ### HOTFIX, if some subjects don't have sufficient data length
        max_len = max(lengths)
        for ts in ts_data:
            ts_data[ts] = [self.pad_array_on_axis_with_nans_to_size(dat, 0, max_len) for dat in ts_data[ts]]
        ts_data = [tc.stack(ts_data[ts]) if len(ts_data[ts])>0 else None for ts in self.timeseries_names]
        return tuple(ts_data)


    def print_dataset_info(self):
        """
        Prints information about each dataset in the wrapper, including its name, 
        total time steps, valid sequences, valid sequence indices, and reference data shape.
        """

        for i, dataset in self.datasets.items():
            log.info(f"Dataset {i} ({dataset.name}):")
            log.info(f"  Total time steps: {len(dataset.reference_data)}")
            log.info(f"  Valid sequences: {len(dataset.valid_sequence_indices)}")
            log.info(f"  Valid sequence indices: {dataset.valid_sequence_indices}")
            log.info(f"  Reference data shape: {dataset.reference_data.shape}")
            log.info("")
    
    def get_nanmean(self):
        nanmeans = []
        for ds in self.datasets:
            nanmeans.append(ds.reference_data.nanmean(axis=0))
        nanmeans = tc.stack(nanmeans, dim=0)
        return nanmeans
    
    def check_seq_len_compatibility(self, dataset_list: list, message_if_failed: str=''):
        seq_len = [d.seq_len for d in dataset_list]
        if len(set(seq_len))>1:
            raise ValueError(message_if_failed)
        
    def check_timeseries_compatibility(self, dataset_list: list, message_if_failed: str=''):
        feature_names = [tuple(d.timeseries.keys()) for d in dataset_list]
        if len(set(feature_names))>1:
            raise ValueError(message_if_failed)
        
    def to(self, device: tc.device):
        for ds in self.datasets:
            ds.to(device)

    @staticmethod
    def pad_array_on_axis_with_nans_to_size(array: tc.Tensor, dim: int, size: int):
        pad_size = list(array.shape)
        pad_size[dim] = max(size - pad_size[dim], 0)
        padding = tc.zeros(pad_size, dtype=array.dtype)
        return tc.cat([array, padding], dim=dim)

class HierarchicalBatchSampler(Sampler):

    def __init__(self, dataset_wrapper: DatasetWrapper, subjects_per_batch: int, seq_per_subject: int, drop_last: bool, shuffle: bool):
        self.dataset_wrapper = dataset_wrapper
        self.subjects_per_batch = subjects_per_batch
        self.seq_per_subject = seq_per_subject
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.available_indices = [(global_idx, local_idx) for global_idx, ds in self.dataset_wrapper.datasets.items() for local_idx in range(len(ds))]
        self.randomize()

    def randomize(self):
        grouped = defaultdict(list)
        for i, j in self.available_indices:
            grouped[i].append(j)
        chunks = []
        for i in grouped:
            random.shuffle(grouped[i])
            while len(grouped[i]) > 0:
                chunks.append([(i, lx) for lx in grouped[i][:self.seq_per_subject]])
                grouped[i] = grouped[i][self.seq_per_subject:]
        random.shuffle(chunks)
        self.chunks = chunks

    def __len__(self):
        if self.drop_last:
            return len(self.dataset_wrapper) // (self.subjects_per_batch * self.seq_per_subject)
        else:
            return ((len(self.dataset_wrapper) + self.subjects_per_batch * self.seq_per_subject - 1) 
                    // (self.subjects_per_batch * self.seq_per_subject))

    def __iter__(self):
        if self.shuffle:
            self.randomize()
        if self.drop_last:
            stop = len(self.chunks) - self.subjects_per_batch + 1
        else:
            stop = len(self.chunks)
        for k in range(0, stop, self.subjects_per_batch):
            batch = self.chunks[k : k+self.subjects_per_batch]
            batch = itertools.chain.from_iterable(batch)
            yield batch