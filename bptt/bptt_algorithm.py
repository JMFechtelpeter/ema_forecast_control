import torch as tc
from torch import optim
from torch import nn
from bptt import plrnn
from bptt import regularization
from bptt import saving
from dataset.multimodal_dataset import MultimodalDataset
from argparse import Namespace
import datetime as dt
from tqdm import tqdm
from timeit import default_timer as timer
import copy

import logging
log = logging.getLogger(__name__)

class BPTT:
    """
    Train a model with (truncated) BPTT.
    """

    def __init__(self, args: Namespace, dataset: MultimodalDataset,
                 validation_dataset: MultimodalDataset,
                  save_path: str, device: tc.device):
        # dataset, model, device, regularizer
        self.device = device
        self.dataset = dataset
        self.val_dataset = validation_dataset
        self.model = plrnn.PLRNN(args, dataset)
        # self.regularizer = regularization.Regularizer(args)
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
        self.saver = saving.Saver( save_path, args, self.dataset, self.val_dataset, None)
        self.model_save_step = args.model_save_step
        self.info_save_step = args.info_save_step
        self.alpha = args.tf_alpha # paramter for teacher forcing
        self.loss_fn = nn.MSELoss()
        self.use_absolute_values_for_loss = args.use_abs_values_for_loss
        self.use_differences_for_loss = args.use_differences_for_loss
        self.data_augmentation = args.data_augmentation
        self.verbose = args.verbose
        self.features = args.dim_x # imp 
        self.participant = args.participant # ?? 
        self.train_on_data_until_datetime = args.train_on_data_until_datetime
        self.validation_len = args.validation_len
        self.validation_prewarming = args.validation_prewarming
        self.save_pred_plots = args.plot_trajectories_after_training
        self.save_loss_plots = args.plot_loss_after_training
        self.early_stopping = args.early_stopping
        self.pbar_descr = args.pbar_descr # ?? 

    def to_device(self) -> None:
        '''
        Moves members to computing device.
        '''
        self.model.to(self.device)
        self.dataset.to(self.device)
        if self.val_dataset is not None:
            self.val_dataset.to(self.device)
        # if self.annealer is not None:
            
        # self.regularizer.to(self.device)

    def compute_loss(self, pred: tc.Tensor, target: tc.Tensor) -> tc.Tensor:
        '''
        Compute Loss w/ optional MAR loss.
        '''
        loss = .0
        # Calculate the mean squared error loss, excluding NaN values from the target
        if self.use_absolute_values_for_loss:
            loss += self.loss_fn(pred[~target.isnan()], target[~target.isnan()]) # excluding the NaN
        # if self.use_reg:
        #     # Get latent parameters from the model
        #     lat_model_parameters = self.model.latent_model.get_latent_parameters()
        #     loss += self.regularizer.loss((lat_model_parameters[0], lat_model_parameters[1], lat_model_parameters[3])) # if A = [0], [W] = [1] why [3]? 
        if self.use_differences_for_loss > 0:
            pred_diff = pred[1:] - pred[:-1]
            target_diff = target[1:] - target[:-1]
            loss += self.loss_fn(pred_diff[~target_diff.isnan()], target_diff[~target_diff.isnan()])


        return loss

    def train(self):

        stopper = EarlyStopper(patience=40)        
        alpha = self.alpha
        epoch_loss_history = []
        val_loss_history = []
        if self.data_augmentation>0:
            noise_dist = tc.distributions.MultivariateNormal(tc.zeros(self.features), 0.2*tc.eye(self.features))    #set up a multivariate normal distribution to sample noise.         
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

        for epoch in epoch_range:

            self.final_epoch = epoch

            # train
            self.model.train()
            epoch_loss = 0
            for batch_count, (emas, inputs) in enumerate(dataloader):
                if batch_count >= self.batches_per_epoch:
                    break
                
                self.optimizer.zero_grad()
                data = emas[:, :-1]
                target = emas[:, 1:]
                if inputs is not None:
                    inputs = inputs[:, :-1]
                if self.data_augmentation>0 and noise_dist is not None: # expand the dataset by replicating the copies of teh original dataset 
                    data_aug = tc.cat([data]*self.data_augmentation, dim=0)
                    noise_ts = noise_dist.sample(data_aug.shape[:2]) #noise shape must be: Batch*Time*Features
                    data_aug = data_aug + noise_ts 
                    data = tc.cat((data, data_aug), dim=0)
                    target = tc.cat([target]*(self.data_augmentation+1), dim=0) # to match teh new size of the input data 
                    if inputs is not None:
                        inputs = tc.cat([inputs]*(self.data_augmentation+1), dim=0)        


                pred, last_z = self.model(data, inputs=inputs, tf_alpha=alpha, return_hidden=True)
                batch_loss = self.compute_loss(pred, target)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                         max_norm=self.gradient_clipping) # to prevent exploding gradients 
                self.optimizer.step()

            # validate
            self.model.eval()
            if self.val_dataset is not None:
                obs, inputs = self.val_dataset.data()
                if self.validation_prewarming > 0:
                    prewarm_obs, prewarm_inputs = self.dataset.data(slice(-self.validation_prewarming-1, -1))
                else:
                    prewarm_obs, prewarm_inputs = None, None
                generated = self.model.generate_free_trajectory(obs[0], obs.shape[0],
                                                                inputs=inputs,
                                                                prewarm_data=prewarm_obs,
                                                                prewarm_inputs=prewarm_inputs,
                                                                prewarm_alpha=alpha)
                validation_target = obs[1:self.validation_len+1]
                generated = generated[:len(validation_target)]
                val_loss = self.compute_loss(generated, validation_target) 
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
                self.saver.save_info(self.model, epoch, epoch_loss, val_loss, self.learning_rate)
            elif epoch > 0 and isinstance(self.model_save_step, int) and epoch % self.model_save_step == 0:     # Saving every kth timestep
                self.saver.save_state_dict(self.model.state_dict(), epoch)
            if epoch > 0 and epoch % self.info_save_step == 0:      # Saving info
                self.saver.save_info(self.model, epoch, epoch_loss, val_loss, self.learning_rate)
                T_end = timer()
                epochs_per_sec = epoch / (T_end-T_start)
                remaining_time = str(dt.timedelta(seconds=round((self.n_epochs - epoch) / epochs_per_sec)))
                if self.annealer is not None:
                    message = f"Epoch {epoch} @ {epochs_per_sec:.1f} epochs/s; epoch/val loss = {epoch_loss:.4f}/{val_loss:.4f}; lr = {self.learning_rate:.6f}; est. {remaining_time} remaining"
                else:
                    message = f"Epoch {epoch} @ {epochs_per_sec:.1f} epochs/s; epoch/val loss = {epoch_loss:.4f}/{val_loss:.4f}; est. {remaining_time} remaining"
                if self.train_on_data_until_datetime not in [None, 'None']:
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