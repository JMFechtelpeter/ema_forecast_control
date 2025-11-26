import os
from argparse import Namespace
import datetime as dt
from tqdm import tqdm
from timeit import default_timer as timer
import torch as tc
from torch import optim
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from hierarchized_bptt.hierarchized_models import HierarchizedPLRNN
from hierarchized_bptt.saving import Saver
from dataset.multimodal_dataset import DatasetWrapper

import logging
log = logging.getLogger(__name__)

class HierarchizedBPTT:
    """
    Train a model with (truncated) BPTT.
    """
    def __init__(self, args: Namespace, dataset_wrapper: DatasetWrapper, validation_dataset_wrapper: DatasetWrapper|None, 
                 save_path: str, device: tc.device):
        # dataset, model, device
        self.device = device
        self.train_wrapper = dataset_wrapper
        self.val_wrapper = validation_dataset_wrapper
        self.model = HierarchizedPLRNN(args, dataset_wrapper=dataset_wrapper, 
                                       load_model_path=args.load_model_path, resume_epoch=args.resume_epoch, new_subjects=args.new_subjects)
        self.to_device()

        self.l2_reg = args.l2_reg
        lr_individual = args.learning_rate_individual + args.individual_lr_bonus_per_subject * self.train_wrapper.n_datasets
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam([{'params': self.model.shared_projection_matrices.values(), 'lr': args.learning_rate},
                                         {'params': self.model.individual_parameters, 'lr': lr_individual}],
                                        weight_decay=self.l2_reg)
        elif args.optimizer == 'RAdam':
            self.optimizer = optim.RAdam([{'params': self.model.shared_projection_matrices.values(), 'lr': args.learning_rate},
                                         {'params': self.model.individual_parameters, 'lr': lr_individual}],
                                        weight_decay=self.l2_reg)
        else:
            raise ValueError('You must choose an optimizer (Adam or RAdam).')

        # others
        # self.subject_index_map = {args.subject_indices[k]: k for k in range(len(args.subject_indices))}
        self.freeze_shared_params = args.freeze_shared_params if args.load_model_path is not None else 0
        self.n_epochs = args.n_epochs
        self.subjects_per_batch = args.subjects_per_batch
        self.seq_per_subject = args.seq_per_subject
        self.batch_size = self.subjects_per_batch * self.seq_per_subject
        self.batches_per_epoch = args.batches_per_epoch
        if self.batches_per_epoch == 0:
            self.batches_per_epoch = len(self.train_wrapper) // self.batch_size + 1
        self.learning_rate = args.learning_rate
        self.gradient_clipping = args.gradient_clipping
        self.saver = Saver(save_path, args, self.train_wrapper, self.val_wrapper)
        self.model_save_step = args.model_save_step
        self.info_save_step = args.info_save_step
        self.alpha = args.tf_alpha # paramter for teacher forcing
        self.loss_fn = nn.MSELoss()
        self.data_augmentation = args.data_augmentation
        self.verbose = args.verbose
        self.features = args.dim_x
        # self.participant = args.participant
        self.train_until = args.train_until
        self.validation_len = args.validation_len
        self.validation_prewarming = args.validation_prewarming
        self.save_pred_plots = args.plot_trajectories_after_training
        self.save_loss_plots = args.plot_loss_after_training
        self.early_stopping = args.early_stopping
        self.pbar_descr = args.pbar_descr

        self.lr_annealing = args.lr_annealing
        if args.lr_annealing=='ReduceLROnPlateau':
            self.annealer = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        elif args.lr_annealing=='ExponentialLR':
            gamma = 0.1 ** (1/args.n_epochs)
            self.annealer = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)
        elif args.lr_annealing=='LinearLR':
            self.annealer = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1., end_factor=0.1, total_iters=self.n_epochs)
        else:
            self.annealer = None


    def to_device(self) -> None:
        self.model.to(self.device)
        self.train_wrapper.to(self.device)        
        if self.val_wrapper is not None:
            self.val_wrapper.to(self.device)

    def compute_loss(self, pred: tc.Tensor, target: tc.Tensor) -> tc.Tensor:
        '''
        Compute Loss w/ optional MAR loss.
        '''
        loss = .0
        # Calculate the mean squared error loss, excluding NaN values from the target
        loss += self.loss_fn(pred[~target.isnan()], target[~target.isnan()])

        return loss

    def train(self):

        stopper = EarlyStopper(patience=self.early_stopping, min_delta=0.05)
        alpha = self.alpha
        epoch_loss_history = []
        val_loss_history = []

        if self.data_augmentation > 0:
            noise_dist = tc.distributions.MultivariateNormal(tc.zeros(self.features), 0.2*tc.eye(self.features))
        else:
            noise_dist = None

        if self.freeze_shared_params:
            self.model.freeze_shared()
        else:
            self.model.unfreeze_shared()

        T_start = timer()
        if self.verbose == 'print':
            epoch_range = range(1, self.n_epochs + 1)
        else:
            epoch_range = tqdm(range(1, self.n_epochs + 1), desc=self.pbar_descr)
        if len(epoch_range)==0:
            raise ValueError('Number of epochs must be greater than 0.')
        if isinstance(self.model_save_step, int):
            log.info('Starting model training. Will save every %i epochs.', self.model_save_step)
        elif self.model_save_step == 'best':
            if self.val_wrapper is not None:
                log.info('Starting model training. Will save model with lowest validation loss.')
            else:
                log.warn('Starting model training. Without a validation set, cannot pick best model. Will save model from last epoch.')
        else:
            log.info('Starting model training. Will save model from last epoch.')

        best_epoch = 0
        best_val_loss = float('inf')

        dataloader = self.train_wrapper.get_dataloader(subjects_per_batch=self.subjects_per_batch,
                                                       seq_per_subject=self.seq_per_subject,
                                                       drop_last=False)
        train_duration = 0
        val_duration = 0
        epoch = 0
        # param_names = self.model.
        
        for epoch in epoch_range:

            # Train
            train_start_time = timer()
            self.model.train()
            epoch_loss = 0
            epoch_gradient_norm = 0
            epoch_param_change = 0

            for batch_count, (subject_idx, (emas, inputs)) in enumerate(dataloader):

                if batch_count >= self.batches_per_epoch:
                    break

                self.optimizer.zero_grad()
                data = emas[:, :-1]
                target = emas[:, 1:]
                if inputs is not None:
                    inputs = inputs[:, :-1]

                if noise_dist is not None:
                    data_aug = tc.cat([data]*self.data_augmentation, dim=0)
                    noise_ts = noise_dist.sample(data_aug.shape[:2])
                    data_aug = data_aug + noise_ts 
                    data = tc.cat((data, data_aug), dim=0)
                    target = tc.cat([target]*(self.data_augmentation+1), dim=0)
                    if inputs is not None:
                        inputs = tc.cat([inputs]*(self.data_augmentation+1), dim=0)

                pred, last_z = self.model(subject_idx, data, inputs=inputs, 
                                            tf_alpha=alpha, return_hidden=True)
                
                batch_loss = self.compute_loss(pred, target)
                if not tc.isnan(batch_loss):
                    epoch_loss += batch_loss.item()
                    batch_loss.backward()
                    grad_norm = clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.gradient_clipping)
                    epoch_gradient_norm += grad_norm.item()
                    parameters_before_optim = [param.clone().detach() for param in self.model.parameters() if param.requires_grad]
                    self.optimizer.step()
                    parameters_after_optim = [param.clone().detach() for param in self.model.parameters() if param.requires_grad]
                    param_change = tc.stack([tc.mean(tc.abs(p_after - p_before)) for p_after, p_before in zip(parameters_after_optim, parameters_before_optim)])
                    epoch_param_change = param_change + epoch_param_change
            epoch_gradient_norm /= len(dataloader)
            epoch_param_change /= len(dataloader)

            # Validate
            val_start_time = timer()
            self.model.eval()
            val_loss = 0.
            if self.val_wrapper is not None:
                # val_obs, val_inputs = tc.zeros((self.validation_len+1, ))
                for idx, val_dataset in self.val_wrapper.datasets.items():
                    train_dataset = self.train_wrapper.datasets[idx]
                    obs, inputs = val_dataset.data()
                    if self.validation_prewarming > 0:
                        prewarm_obs, prewarm_inputs = train_dataset.data(slice(-self.validation_prewarming-1, -1))
                    else:
                        prewarm_obs, prewarm_inputs = None, None
                    generated = self.model.generate_free_trajectory(idx, obs[0], obs.shape[0],
                                                                    inputs=inputs,
                                                                    prewarm_data=prewarm_obs,
                                                                    prewarm_inputs=prewarm_inputs,
                                                                    prewarm_alpha=alpha)
                    validation_target = obs[1:self.validation_len+1]
                    generated = generated[:len(validation_target)]
                    subject_val_loss = self.loss_fn(generated[~validation_target.isnan()], validation_target[~validation_target.isnan()])
                    if not tc.isnan(subject_val_loss):
                        val_loss += subject_val_loss.item()

                val_loss /= len(self.val_wrapper)

            else:
                val_loss = 0.
            val_end_time = timer()
            train_duration += val_start_time - train_start_time
            val_duration += val_end_time - val_start_time

            if self.l2_reg > 0:
                l2_loss = self.l2_reg * sum([tc.sum(param**2) for param in self.model.parameters() if param.requires_grad])
            else:
                l2_loss = 0.

            # Learning rate annealing
            if self.annealer is not None:
                if self.lr_annealing == 'ReduceLROnPlateau':
                    self.annealer.step(val_loss)
                else:
                    self.annealer.step()
            self.learning_rate = self.optimizer.param_groups[0]['lr']
            self.learning_rate_individual = self.optimizer.param_groups[1]['lr']

            epoch_loss /= self.batches_per_epoch
            epoch_loss_history.append(epoch_loss)
            val_loss_history.append(val_loss)

            # Model saving logic
            if self.model_save_step == 'best' and val_loss < best_val_loss:
                best_model_state_dict = self.model.state_dict()
                best_val_loss = val_loss
                best_epoch = epoch
                self.saver.save_info(self.model, epoch, epoch_loss, val_loss, l2_loss,
                                     self.learning_rate, self.learning_rate_individual, 
                                     epoch_gradient_norm, epoch_param_change)
            elif epoch > 0 and isinstance(self.model_save_step, int) and epoch % self.model_save_step == 0:
                self.saver.save_state_dict(self.model.state_dict(), epoch)
                self.saver.save_info(self.model, epoch, epoch_loss, val_loss, l2_loss,
                                     self.learning_rate, self.learning_rate_individual, 
                                     epoch_gradient_norm, epoch_param_change)
                self.log_progress(epoch, epoch_loss, T_start)
            # Info saving
            elif epoch > 0 and epoch % self.info_save_step == 0:
                self.saver.save_info(self.model, epoch, epoch_loss, val_loss, l2_loss,
                                     self.learning_rate, self.learning_rate_individual, 
                                     epoch_gradient_norm, epoch_param_change)
                
            # Early stopping check
            if self.early_stopping and stopper.decide_stop(val_loss, self.learning_rate):
                log.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Final model saving and plotting
        if self.model_save_step == 'best':
            if tc.isnan(tc.tensor(val_loss_history)).all():
                self.saver.save_state_dict(self.model.state_dict(), epoch)
                log.info('Saved last model, because validation error was always NaN.')
            else:
                self.saver.save_state_dict(best_model_state_dict, best_epoch)
                log.info(f'Saved best model (best_epoch={best_epoch}).')
        elif self.model_save_step == 'last':
            self.saver.save_state_dict(self.model.state_dict(), epoch)
            log.info('Saved model from last epoch.')
        
        if self.save_loss_plots:
            self.saver.plot_loss()
        if self.save_pred_plots:
            self.saver.save_plots()

        log.info(f'Training took {train_duration:.2f} seconds; validation took {val_duration:.2f} seconds.')
    
    def log_progress(self, epoch, epoch_loss, T_start):

        T_end = timer()
        epochs_per_sec = epoch / (T_end - T_start)
        remaining_time = str(dt.timedelta(seconds=round((self.n_epochs - epoch) / epochs_per_sec)))
        if self.annealer is not None:
            message = f"Epoch {epoch} @ {epochs_per_sec:.1f} epochs/s; loss = {epoch_loss:.4f}; lr = {self.learning_rate:.6f}; est. {remaining_time} remaining"
        else:
            message = f"Epoch {epoch} @ {epochs_per_sec:.1f} epochs/s; loss = {epoch_loss:.4f}; est. {remaining_time} remaining"
        if self.train_until is not None:
            message = f'Test split @ {self.train_until}; ' + message
        log.info(message)


class EarlyStopper:
    def __init__(self, patience: int=20, min_delta: float=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def decide_stop(self, validation_loss: float, learning_rate: float):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta*learning_rate*1e3):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False