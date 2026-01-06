import logging
log = logging.getLogger(__name__)

from typing import Optional
import os
from timeit import default_timer as timer
import datetime as dt

import torch as tc
from torch import optim
from torch import nn
import pandas as pd
from tqdm import tqdm

from ema_forecast_control.dataset.time_series_dataset import TimeSeriesDataset
from ema_forecast_control.plrnn.plrnn_model import PLRNN

class BPTT:

    def __init__(self, args: dict, dataset: TimeSeriesDataset, test_data: Optional[tc.Tensor], test_inputs: Optional[tc.Tensor], 
                 save_path: str, device: tc.device):

        self.device = device
        self.dataset = dataset
        self.test_data = test_data
        self.test_inputs = test_inputs
        self.model = PLRNN(args, dataset)
        self.to_device()

        self.optimizer = optim.Adam(self.model.parameters(), args['learning_rate'])
        if args['lr_annealing']:
            self.annealer = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        else:
            self.annealer = None
        
        # others
        self.n_epochs = args['n_epochs']
        self.batch_size = args['batch_size']
        self.batches_per_epoch = args['batches_per_epoch']
        if self.batches_per_epoch == 0:
            self.batches_per_epoch = len(self.dataset) // self.batch_size
        self.learning_rate = args['learning_rate']
        self.gradient_clipping = args['gradient_clipping']
        self.model_save_step = args['model_save_step']
        self.info_save_step = args['info_save_step']
        self.alpha = args['tf_alpha']
        self.loss_fn = nn.MSELoss()
        self.verbose = args['verbose']
        self.features = args['dim_x']
        self.early_stopping = args['early_stopping']
        self.pbar_descr = args['pbar_descr']
        self.validation_len = args['validation_len']
        self.validation_prewarming = args['validation_prewarming']
        self.loss_df = None
        self.save_path = save_path

    def to_device(self) -> None:
        self.model.to(self.device)
        self.dataset.to(self.device)
        if self.test_data is not None:
            self.test_data.to(self.device)
        if self.test_inputs is not None:
            self.test_inputs.to(self.device)

    def compute_loss_excluding_missings(self, pred: tc.Tensor, target: tc.Tensor) -> tc.Tensor:
        return self.loss_fn(pred[~target.isnan()], target[~target.isnan()])
    
    def train(self):

        stopper = EarlyStopper(patience=40)        
        alpha = self.alpha
        epoch_loss_history = []
        val_loss_history = []
        T_start = timer()
        if self.verbose == 'print':
            epoch_range = range(1, self.n_epochs + 1)
        else:
            epoch_range = tqdm(range(1, self.n_epochs + 1), desc=self.pbar_descr)
        if isinstance(self.model_save_step, int):
            log.info('Starting model training. Will save every %i epochs.', self.model_save_step)
        elif self.model_save_step == 'best':
            if self.test_data is not None:
                log.info('Starting model training. Will save model with lowest validation loss.')
            else:
                log.warn('Starting model training. Without a validation set, cannot pick best model. Will save model from last epoch.')
        else:
            log.info('Starting model training. Will save model from last epoch.')
        
        dataloader = self.dataset.get_dataloader(self.batch_size, shuffle=True, drop_last=True)
        best_epoch = None
        best_model_state_dict = None

        for epoch in epoch_range:

            self.final_epoch = epoch

            # train
            self.model.train()
            epoch_loss = 0
            for batch_count, (data, inputs) in enumerate(dataloader):
                if batch_count >= self.batches_per_epoch:
                    break
                
                self.optimizer.zero_grad()
                target = data[:, 1:]
                data = data[:, :-1]                
                if inputs is not None:
                    inputs = inputs[:, :-1]

                pred, last_z = self.model(data, inputs=inputs, tf_alpha=alpha, return_hidden=True)
                batch_loss = self.compute_loss_excluding_missings(pred, target)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                         max_norm=self.gradient_clipping) # to prevent exploding gradients 
                self.optimizer.step()

            # validate
            self.model.eval()
            if self.test_data is not None:
                if self.validation_prewarming > 0:
                    prewarm_data = self.dataset.data(slice(-self.validation_prewarming-1, -1))
                    if self.dataset.inputs is not None:
                        prewarm_inputs = self.dataset.inputs(slice(-self.validation_prewarming-1, -1))
                else:
                    prewarm_data, prewarm_inputs = None, None
                generated = self.model.generate_free_trajectory(self.test_data[0], self.test_data.shape[0], inputs=self.test_inputs,
                                                                prewarm_data=prewarm_data, prewarm_inputs=prewarm_inputs, prewarm_alpha=alpha)
                validation_target = self.test_data[1:self.validation_len+1]
                generated = generated[:len(validation_target)]
                val_loss = self.compute_loss_excluding_missings(generated, validation_target) 
            else:
                val_loss = 0.

            # anneal learning rate
            if self.annealer is not None:
                self.annealer.step(val_loss)
                self.learning_rate = self.optimizer.param_groups[0]['lr']
                                   
            epoch_loss /= self.batches_per_epoch
            epoch_loss_history.append(epoch_loss)
            val_loss_history.append(val_loss)
            if self.model_save_step=='best' and val_loss == min(val_loss_history):  # Tracking best model
                best_model_state_dict = self.model.state_dict()
                best_epoch = epoch
                self.track_training(epoch, epoch_loss, val_loss, self.learning_rate)
            elif epoch > 0 and isinstance(self.model_save_step, int) and epoch % self.model_save_step == 0:     # Saving every kth timestep
                tc.save(self.model.state_dict(), os.path.join(self.save_path, f'model_{epoch}.pt'))
            if epoch > 0 and epoch % self.info_save_step == 0:      # Saving info
                self.track_training(epoch, epoch_loss, val_loss, self.learning_rate)
                T_end = timer()
                epochs_per_sec = epoch / (T_end-T_start)
                remaining_time = str(dt.timedelta(seconds=round((self.n_epochs - epoch) / epochs_per_sec)))
                if self.annealer is not None:
                    message = f"Epoch {epoch} @ {epochs_per_sec:.1f} epochs/s; epoch/val loss = {epoch_loss:.4f}/{val_loss:.4f}; lr = {self.learning_rate:.6f}; est. {remaining_time} remaining"
                else:
                    message = f"Epoch {epoch} @ {epochs_per_sec:.1f} epochs/s; epoch/val loss = {epoch_loss:.4f}/{val_loss:.4f}; est. {remaining_time} remaining"
                log.info(message)

            if self.early_stopping and stopper.decide_stop(val_loss, self.learning_rate):
                break
        
        if self.model_save_step=='best':
            if tc.isnan(tc.tensor(val_loss_history)).all():
                tc.save(self.model.state_dict(), os.path.join(self.save_path, f'model_{self.final_epoch}.pt'))
                log.info('Saved last model, because validation error was always NaN.')
            elif best_epoch is not None and best_model_state_dict is not None:
                tc.save(best_model_state_dict, os.path.join(self.save_path, f'model_{best_epoch}.pt'))
                log.info('Saved best model (best_epoch=%i).', best_epoch)
            else:
                tc.save(self.model.state_dict(), os.path.join(self.save_path, f'model_{self.final_epoch}.pt'))
                log.info('Saved last model, because no best model was found.')
        elif self.model_save_step=='last':
            tc.save(self.model.state_dict(), os.path.join(self.save_path, f'model_{self.final_epoch}.pt'))
            log.info('Saved model from last epoch.')

    def track_training(self, epoch, epoch_loss, validation_loss, learning_rate):

        with tc.no_grad():
            
            model_parameters = self.model.get_parameters()

            # Keep in mind: We clip the gradients from the last backward pass of the training loop at
            # current epoch here, which are already clipped during training
            # so this line has the sole purpose of getting the total_norm from the last gradients
            total_grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.gradient_clipping)
            loss_df = pd.DataFrame(index=[epoch])
            if isinstance(epoch_loss, tc.Tensor):
                loss_df['epoch_loss'] = epoch_loss.item()
            else:
                loss_df['epoch_loss'] = epoch_loss
            if isinstance(validation_loss, tc.Tensor):
                loss_df['validation_loss'] = validation_loss.item()
            else:
                loss_df['validation_loss'] = validation_loss
            loss_df['learning_rate'] = learning_rate
            loss_df['L2_norm_A'] = tc.linalg.norm(tc.diag(model_parameters['A']), 2).item()
            if model_parameters['C'] is not None:
                loss_df['L2_norm_C'] = tc.linalg.norm(model_parameters['C'], 2).item()
            loss_df['total_grad_norm'] = total_grad_norm.item()
            if self.loss_df is None:
                self.loss_df = loss_df
            else:
                self.loss_df = pd.concat((self.loss_df, loss_df))          
            self.loss_df.to_csv(os.path.join(self.save_path, 'loss.csv'))


class EarlyStopper:
    def __init__(self, patience: int=20, min_delta: float=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def decide_stop(self, validation_loss: float, learning_rate: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta*learning_rate*1e3):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False