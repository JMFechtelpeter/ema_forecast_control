import sys, os
from timeit import default_timer as timer
import copy
import datetime as dt
from tqdm import tqdm
from argparse import Namespace
from typing import Optional

import torch as tc
from torch import optim
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from comparison_models.transformer import saving
from comparison_models.transformer import time_series_transformer
try:
    sys.path.append('../..')
    from dataset.multimodal_dataset import MultimodalDataset
except:
    sys.path.append(os.getcwd())
    from dataset.multimodal_dataset import MultimodalDataset 

import logging
log = logging.getLogger(__name__)

class TransformerTrainer:

    def __init__(self, args: dict|Namespace, dataset: MultimodalDataset,
                 validation_dataset: Optional[MultimodalDataset],
                 writer, save_path: str, device: tc.device):
        
        if isinstance(args, dict):
            args = Namespace(**args)

        # dataset, model, device, regularizer, 
        self.device = device
        self.dataset = dataset
        self.val_dataset = validation_dataset
        self.model = time_series_transformer.AutoregressiveTransformer(args)
        self.to_device()

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), args.learning_rate)
        if args.lr_annealing:
            self.annealer = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
            # self.annealer = optim.lr_scheduler.CyclicLR(self.optimizer, 1e-5, args.learning_rate, 
            #                                             step_size_up=500, mode='triangular2', cycle_momentum=False)
            # self.annealer = optim.lr_scheduler.StepLR(self.optimizer, 100, 0.8)
        else:
            self.annealer = None
        
        # others
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.batches_per_epoch = args.batches_per_epoch
        if self.batches_per_epoch == 0:
            self.batches_per_epoch = len(self.dataset) // self.batch_size
        self.learning_rate = args.learning_rate
        self.gradient_clipping = args.gradient_clipping
        self.writer = writer
        self.saver = saving.Saver(writer, save_path, args, self.dataset, self.val_dataset)
        self.model_save_step = args.model_save_step
        self.info_save_step = args.info_save_step
        self.loss_fn = nn.MSELoss()
        self.data_augmentation = args.data_augmentation
        self.verbose = args.verbose
        self.features = args.dim_x
        self.participant = args.participant
        self.decoder_seq_len = args.decoder_seq_len
        self.train_on_data_until_datetime = args.train_on_data_until_datetime
        self.validation_len = args.validation_len
        self.validation_prewarming = args.validation_prewarming
        self.save_pred_plots = args.plot_trajectories_after_training
        self.save_loss_plots = args.plot_loss_after_training
        self.early_stopping = args.early_stopping
        self.pbar_descr = args.pbar_descr

        # print(self.model.count_parameters())

    def to_device(self) -> None:
        self.model.to_device(self.device)
        self.dataset.to(self.device)
        if self.val_dataset is not None:
            self.val_dataset.to(self.device)

    def compute_loss(self, pred: tc.Tensor, target: tc.Tensor) -> tc.Tensor:
        return self.loss_fn(pred[~target.isnan()], target[~target.isnan()])

    def train(self):

        stopper = EarlyStopper(patience=40)
        epoch_loss_history = []
        val_loss_history = []
        if self.data_augmentation>0:
            noise_dist = tc.distributions.MultivariateNormal(tc.zeros(self.features), 0.2*tc.eye(self.features))            
        else:
            noise_dist = None
        T_start = timer()
        if self.verbose == 'print':
            epoch_range = range(1, self.n_epochs + 1)
        else:
            epoch_range = tqdm(range(1, self.n_epochs + 1), desc=self.pbar_descr)
        if isinstance(self.model_save_step, int):
            log.info('Starting model training. Will save every %i epochs.', self.model_save_step)
        elif self.model_save_step == 'best':
            if self.val_dataset is not None:
                log.info('Starting model training. Will save model with lowest validation loss.')
            else:
                log.warn('Starting model training. Without a validation set, cannot pick best model. Will save model from last epoch.')
        else:
            log.info('Starting model training. Will save model from last epoch.')

        dataloader = self.dataset.get_dataloader(self.batch_size, shuffle=True, drop_last=True)
        best_epoch = None
        best_model_state_dict = None
        self.final_epoch = 0
        dsl = self.decoder_seq_len
        for epoch in epoch_range:
            self.final_epoch = epoch
            self.model.train()
            epoch_loss = 0
            for batch_count, (emas, inputs) in enumerate(dataloader):   # shape batch*time*features
                if batch_count >= self.batches_per_epoch:
                    break

                self.optimizer.zero_grad()
                b, t, dz = emas.shape
                if self.data_augmentation>0 and noise_dist is not None:
                    emas_aug = tc.cat([emas]*self.data_augmentation, dim=0)
                    noise_ts = noise_dist.sample(emas_aug.shape[:2]) #noise shape must be: Batch*Time*Features
                    emas_aug = emas_aug + noise_ts
                    emas = tc.cat((emas, emas_aug), dim=0)
                    if inputs is not None:
                        inputs = tc.cat([inputs]*(self.data_augmentation+1), dim=0)  
                
                # if inputs is not None:
                #     combined_input = tc.cat([emas, inputs], dim=-1)
                # else:
                #     combined_input = emas
                # src = combined_input[:, :-dsl]
                # tgt = combined_input[:, -dsl-1:-1]
                target = emas[:, -dsl:]
                # src_mask = self.model.get_decoder_mask(dsl, t-dsl)
                # tgt_mask = self.model.get_decoder_mask(dsl, dsl)                      
                pred = self.model(emas, inputs=inputs, mask_past=True)
                batch_loss = self.compute_loss(pred, target)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                clip_grad_norm_(parameters=self.model.parameters(),
                                         max_norm=self.gradient_clipping)
                self.optimizer.step()

            # validate
            self.model.eval()
            if self.val_dataset is not None:
                obs, inputs = self.val_dataset.data()
                if self.validation_prewarming > 0:
                    prewarm_obs, prewarm_inputs = self.dataset.data(slice(-self.validation_prewarming-1, -1))
                else:
                    prewarm_obs, prewarm_inputs = None, None
                generated, _ = self.model.generate_free_trajectory(obs[0], obs.shape[0], inputs,
                                                                    prewarm_data=prewarm_obs,
                                                                    prewarm_inputs=prewarm_inputs)
                validation_target = obs[1:self.validation_len+1]
                generated = generated[:len(validation_target)]
                val_loss = self.loss_fn(generated[~validation_target.isnan()], validation_target[~validation_target.isnan()]) 
            else:
                val_loss = 0.

            # anneal learning rate
            if self.annealer is not None:
                self.annealer.step(val_loss)
                self.learning_rate = self.optimizer.param_groups[0]['lr']
                                   
            epoch_loss /= self.batches_per_epoch
            epoch_loss_history.append(epoch_loss)
            val_loss_history.append(val_loss)
            if self.model_save_step=='best' and val_loss == min(val_loss_history):
                best_model_state_dict = self.model.state_dict()
                best_epoch = epoch
                self.saver.save_info(self.model, epoch, epoch_loss, val_loss)
            elif epoch > 0 and isinstance(self.model_save_step, int) and epoch % self.model_save_step == 0:
                self.saver.save_state_dict(self.model.state_dict(), epoch)
            if epoch > 0 and epoch % self.info_save_step == 0:
                self.saver.save_info(self.model, epoch, epoch_loss, val_loss)
                T_end = timer()
                epochs_per_sec = epoch / (T_end-T_start)
                remaining_time = str(dt.timedelta(seconds=round((self.n_epochs - epoch) / epochs_per_sec)))
                if self.annealer is not None:
                    message = f"Epoch {epoch} @ {epochs_per_sec:.1f} epochs/s; loss = {epoch_loss:.4f}; lr = {self.learning_rate:.6f}; est. {remaining_time} remaining"
                else:
                    message = f"Epoch {epoch} @ {epochs_per_sec:.1f} epochs/s; loss = {epoch_loss:.4f}; est. {remaining_time} remaining"
                if self.train_on_data_until_datetime is not None:
                    message = f'Date {self.train_on_data_until_datetime}; ' + message
                log.info(message)

            if self.early_stopping and stopper.decide_stop(val_loss, self.learning_rate):
                break
        
        if self.model_save_step=='best':
            if tc.isnan(tc.tensor(val_loss_history)).all():
                self.saver.save_state_dict(self.model.state_dict(), self.final_epoch)
                log.info('Saved last model, because validation error was always NaN.')
            elif best_epoch is not None and best_model_state_dict is not None:
                self.saver.save_state_dict(best_model_state_dict, best_epoch)
                log.info('Saved best model (best_epoch=%i).', best_epoch)
            else:
                self.saver.save_state_dict(self.model.state_dict(), self.final_epoch)
                log.info('Saved last model, because no best model was found.')
        elif self.model_save_step=='last':
            self.saver.save_state_dict(self.model.state_dict(), self.final_epoch)
            log.info('Saved model from last epoch.')
        if self.save_loss_plots:
            self.saver.plot_loss()
        if self.save_pred_plots:
            self.saver.save_plots()

        self.model.eval()   


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
