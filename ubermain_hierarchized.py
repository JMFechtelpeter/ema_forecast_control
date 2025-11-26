# from multitasking import Argument, ArgumentCombination, ArgumentTable, run_settings, create_tasks_from_arguments
# import os
# import time
# import shutil
# from data_utils import join_base_path, dataset_path, train_test_split_path

# MRT = 1

# def ubermain(n_runs):
#     """
#     Specify the argument choices you want to be tested here in list format:
#     e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
#     will test for dimensions 5 and 6 and save experiments under z5 and z6
#     """
#     args = []
#     args.append(Argument('experiment', ['hiera_gridsearch_10splits_dim_p']))
#     args.append(Argument('name', ['model']))
#     args.append(Argument('use_gpu', [0]))
#     args.append(Argument('use_tb', [0]))
#     args.append(Argument('plot_trajectories_after_training', [0]))
#     args.append(Argument('plot_loss_after_training', [0]))
#     args.append(Argument('n_epochs', [300]))
#     args.append(Argument('model_save_step', ['best']))
#     args.append(Argument('info_save_step', [10]))
#     args.append(Argument('tf_alpha', [0.125]))#, add_to_name_as='alpha'))
#     args.append(Argument('dim_z', [6]))#, add_to_name_as='dim_z'))
#     args.append(Argument('dim_y', [15]))#, add_to_name_as='dim_y'))
#     args.append(Argument('dim_x_proj', [6]))#, add_to_name_as='dim_x_proj'))
#     # args.append(ArgumentCombination(('subjects_per_batch', 'seq_per_subject'), 
#     #                                 [(4,4), (6,6), (8,8), (16,8)],
#     #                                 add_to_name_as=('subj_per_batch', 'seq_per_subj')))
#     args.append(Argument('dim_p', [2, 3, 4, 6, 8, 10, 12, 14], add_to_name_as='dim_p'))
#     args.append(Argument('seq_len', [7]))#, add_to_name_as='seq_len'))
#     args.append(Argument('subjects_per_batch', [8]))
#     args.append(Argument('seq_per_subject', [8]))
#     args.append(Argument('batches_per_epoch', [0]))
#     args.append(Argument('learning_rate', [1e-3]))
#     args.append(Argument('lr_annealing', [1]))#, add_to_name_as='annealing'))
#     args.append(Argument('train_on_last_n_steps', [None]))# [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140], add_to_name_as='train_length'))
#     args.append(Argument('run', list(range(1, 1 + n_runs))))    
#     args.append(Argument('data_path', [dataset_path(MRT, 'processed_csv_no_con_smoothed')]))
#     args.append(Argument('participant_subset_selector', [train_test_split_path(MRT, 'balanced_validation_no_con_smoothed_5_splits_after_80.csv')]))
#     args.append(Argument('train_until', [train_test_split_path(MRT, 'balanced_validation_no_con_smoothed_5_splits_after_80.csv')]))
#     args.append(Argument('train_test_split_row', [0, 1, 2, 3, 4], add_to_name_as='splitrow'))


#     return args


# if __name__ == '__main__':
#     os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#     # number of runs for each experiment
#     n_runs = 10
#     # number of runs to run in parallel
#     n_cpu = 50
#     # number of processes run parallel on a single GPU
#     n_proc_per_gpu = 1

#     args = ubermain(n_runs)

#     tasks, n_cpu = create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu, main_path='main_hierarchized.py')
#     print(f'Running {len(tasks)} jobs, {n_cpu} in parallel. Proceeds in 10 seconds.')
#     time.sleep(10)
#     run_settings(tasks, n_cpu)
#     shutil.copy('ubermain.py', os.path.join(os.getcwd(), f'results/{args[0].name}'))




from typing import Optional
import os
import re
import time
import shutil
import pandas as pd
from multitasking import Argument, ArgumentCombination, ArgumentTable, run_settings, create_tasks_from_arguments
from data_utils import join_ordinal_bptt_path, dataset_path, train_test_split_path
import utils


def filter_participants(data_dir: str, participants: Optional[list]) -> list:
    data_list = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            if participants is not None:
                re_result = re.search('_([0-9]+).csv', file)
                if re_result is not None:
                    participant = re_result.group(1)
                    if participant in participants or int(participant) in participants:
                        data_list.append(os.path.join(data_dir, file))
            else:
                data_list.append(os.path.join(data_dir, file))
    return data_list

def filter_participants_according_to_test_splits(data_dir: str, test_split_file: str) -> list:
    use_participants = pd.read_csv(test_split_file, index_col=0).columns.to_list()
    data_list = filter_participants(data_dir, use_participants)
    return data_list


def ubermain(hyperparameters: list, n_runs: int, results_folder: str, experiment: str, 
             data_dir: str, n_cpu: int, n_proc_per_gpu: int, 
             main_path: str=join_ordinal_bptt_path('main.py')):

    args = []

    args.append(Argument('results_folder', [results_folder]))
    args.append(Argument('experiment', [experiment]))
    args.append(Argument('name', ['model']))
    args.append(Argument('run', list(range(1, 1 + n_runs))))    
    args.append(Argument('data_path', [data_dir]))

    experiment_path = utils.get_experiment_path(results_folder, experiment)
    if os.path.exists(experiment_path):
        answer = input(f'Experiment path {experiment_path} already exists. Delete/Overwrite/Update/Abort? (d/o/u/a)')
        if answer=='d':
            shutil.rmtree(experiment_path)
        elif answer=='o':
            args.append(Argument('overwrite', [1]))
        elif answer=='u':
            args.append(Argument('overwrite', [0]))
        else:
            return
        
    args.extend(hyperparameters)

    tasks, n_cpu = create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu, main_path=main_path)
    print(f'Running {len(tasks)} jobs, {n_cpu} in parallel. Proceeds in 10 seconds.')
    time.sleep(10)
    os.makedirs(experiment_path, exist_ok=True)
    shutil.copyfile(join_ordinal_bptt_path('ubermain.py'), os.path.join(experiment_path, 'ubermain.py'))
    run_settings(tasks, n_cpu)
    

if __name__ == '__main__':

    MRT = 3

    results_folder = 'results'
    experiment = 'hiera_MRT3_every_day'
    data_dir = dataset_path(MRT, 'processed_csv_no_con_smoothed')
    list_of_participants = None
    n_runs = 10
    n_cpu = 30
    n_proc_per_gpu = 1

    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6
    """

    args = []
    args.append(Argument('early_stopping', [1]))
    args.append(Argument('use_gpu', [0]))
    args.append(Argument('plot_trajectories_after_training', [0]))
    args.append(Argument('plot_loss_after_training', [1]))
    args.append(Argument('n_epochs', [200]))
    args.append(Argument('model_save_step', ['best']))
    args.append(Argument('info_save_step', [2]))
    args.append(Argument('tf_alpha', [0.125]))#, add_to_name_as='alpha'))
    args.append(Argument('dim_y', [15]))#, add_to_name_as='dim_y'))
    args.append(Argument('dim_z', [6]))#, add_to_name_as='dim_z'))
    args.append(Argument('dim_x_proj', [6]))#, add_to_name_as='dim_x_proj'))
    args.append(Argument('dim_p', [12]))
    # args.append(ArgumentCombination(('dim_z', 'dim_x_proj', 'dim_p'), 
    #                                 [(6, 6, 4), (6, 6, 8), (6, 6, 12), 
    #                                  (12, 12, 4), (12, 12, 8), (12, 12, 12)],
    #                                 add_to_name_as=('dim_z', 'dim_x_proj', 'dim_p')))
    args.append(Argument('seq_len', [12]))#, add_to_name_as='seq_len'))
    args.append(Argument('subjects_per_batch', [16]))#, add_to_name_as='subj_per_batch'))
    args.append(Argument('seq_per_subject', [8]))#, add_to_name_as='seq_per_subj'))
    args.append(Argument('batches_per_epoch', [0]))
    args.append(Argument('learning_rate', [1e-3]))
    args.append(Argument('learning_rate_individual', [1e-3]))
    args.append(Argument('individual_lr_bonus_per_subject', [1e-2 / 60]))
    # args.append(ArgumentCombination(('learning_rate', 'learning_rate_individual', 'individual_lr_bonus_per_subject'), 
    #                                 [(1e-3, 1e-3, 0), (1e-3, 1e-3, 1e-2 / 60), (1e-3, 1e-2, 0),
    #                                  (5e-4, 5e-4, 0), (5e-4, 5e-4, 5e-3 / 60), (5e-4, 5e-3, 0),
    #                                  (1e-4, 1e-4, 0), (1e-4, 1e-4, 1e-3 / 60), (1e-4, 1e-3, 0)],
                                    # add_to_name_as=('lr', 'lr_ind', 'lr_bonus')))
    args.append(Argument('lr_annealing', ['ReduceLROnPlateau']))#, add_to_name_as='annealing'))
    args.append(Argument('optimizer', ['Adam']))#, add_to_name_as='optim'))
    args.append(Argument('l2_reg', [0]))#, 1e-5, 1e-4], add_to_name_as='reg'))
    args.append(Argument('train_on_last_n_steps', [None]))# [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140], add_to_name_as='train_length'))
    args.append(Argument('participant_subset_selector', [train_test_split_path(MRT, 'valid_first_alarms_no_con.csv')]))#[train_test_split_path(MRT, 'balanced_validation_no_con_smoothed_5_splits_after_80.csv')]))
    args.append(Argument('train_until', pd.read_csv(train_test_split_path(MRT, 'dates_of_valid_first_alarms_no_con.csv'), index_col=0)['0'].to_list(), add_to_name_as='train_until'))
    # args.append(Argument('train_until', pd.date_range('2022-08-01', '2023-03-01').strftime('%Y-%m-%d').to_list(), add_to_name_as='train_until')) # for MRT 1
    # args.append(Argument('train_until', pd.date_range('2023-04-01', '2024-02-01').strftime('%Y-%m-%d').to_list(), add_to_name_as='train_until')) # for MRT 2
    # args.append(Argument('train_until', pd.date_range('2023-11-01', '2024-08-01').strftime('%Y-%m-%d').to_list(), add_to_name_as='train_until')) # for MRT 3
    # args.append(Argument('train_until', [train_test_split_path(MRT, 'valid_first_alarms_no_con.csv')]))
    # args.append(Argument('train_test_split_row', list(range(27)), add_to_name_as='splitrow'))

    # args.append(ArgumentTable('load_model_path', 'run', table=pd.DataFrame({str(run): [join_ordinal_bptt_path('results/hiera_MRT1_all_data', str(run).zfill(3)) ]
    #                                                                         for run in range(1, n_runs+1)}),
    #                           add_to_name_as='run'))
    # args.append(Argument('new_subjects', ['replace']))
    args.append(Argument('freeze_shared_params', [0]))

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    ubermain(args, n_runs, results_folder, experiment, data_dir, 
             n_cpu, n_proc_per_gpu,
             join_ordinal_bptt_path('main_hierarchized.py'))
