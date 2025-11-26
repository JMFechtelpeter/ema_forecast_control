from typing import Optional
import os, sys
import re
import time
import shutil
import pandas as pd
if __name__=='__main__':
    sys.path.append(os.getcwd())
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
             data_dir: str, test_split_file: Optional[str], list_of_participants: Optional[list],
             n_cpu: int, n_proc_per_gpu: int):
    
    assert os.path.exists(data_dir), f'Data directory {data_dir} does not exist.'
    if test_split_file is not None:
        assert os.path.exists(test_split_file), f'Test split file {test_split_file} does not exist.'

    if list_of_participants is None and test_split_file is not None:
        data_list = filter_participants_according_to_test_splits(data_dir, test_split_file)
    else:
        data_list = filter_participants(data_dir, list_of_participants)
    args = []

    ### MANDATORY ARGUMENTS
    args.append(Argument('results_folder', [results_folder]))
    args.append(Argument('experiment', [experiment]))
    args.append(Argument('name', ['model']))
    args.append(Argument('run', list(range(1, 1 + n_runs))))    
    args.append(Argument('data_path', data_list, add_to_name_as='data'))
    args.append(Argument('participant', [], add_to_name_as='participant',
                         infer_value_from='data_path', infer_value_pattern='--data_path .+?([0-9]+).csv'))
    if test_split_file is not None:
        args.append(ArgumentTable('train_on_data_until_timestep', 'participant',
                                file_path=test_split_file,
                                add_to_name_as='date'))
    
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

    tasks, n_cpu = create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu, main_path=join_ordinal_bptt_path('comparison_models/transformer/main_transformer.py'))
    print(f'Running {len(tasks)} jobs, {n_cpu} in parallel. Proceeds in 10 seconds.')
    time.sleep(10)    
    os.makedirs(experiment_path, exist_ok=True)
    shutil.copyfile(join_ordinal_bptt_path('comparison_models/transformer/ubermain_transformer.py'), os.path.join(experiment_path, 'ubermain_transformer.py'))
    run_settings(tasks, n_cpu)
    



if __name__ == '__main__':

    MRT = 3

    results_folder = 'results'
    experiment = 'v3_MRT3_Transformer_every_valid_day_seq_len_7'
    data_dir = dataset_path(MRT, 'processed_csv_no_con_smoothed_imputed_causal')
    test_split_file = train_test_split_path(MRT, 'valid_first_alarms_no_con_smoothed.csv')
    list_of_participants = None
    n_runs = 10
    n_cpu = 50
    n_proc_per_gpu = 1

    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6
    """

    args = []
    args.append(Argument('use_gpu', [0]))
    args.append(Argument('use_tb', [0]))
    args.append(Argument('plot_trajectories_after_training', [0]))
    args.append(Argument('plot_loss_after_training', [0]))
    args.append(Argument('n_epochs', [1200]))
    args.append(Argument('model_save_step', ['best']))
    args.append(Argument('info_save_step', [10]))
    args.append(Argument('dim_model', [4]))#, add_to_name_as='dim_model'))
    args.append(Argument('dim_feedforward', [5]))#, add_to_name_as='dim_ff'))
    args.append(Argument('decoder_seq_len', [6]))#, add_to_name_as='dsl'))
    args.append(Argument('n_encoder_layers', [1]))#, add_to_name_as='n_enc_layers'))
    args.append(Argument('n_decoder_layers', [1]))#, add_to_name_as='n_dec_layers'))
    # args.append(ArgumentCombination(('n_encoder_layers', 'n_decoder_layers'), 
    #                                 [(1,1), (2,2)],
    #                                 add_to_name_as=('n_enc_layers', 'n_dec_layers')))
    args.append(Argument('n_heads', [1]))#, add_to_name_as='n_heads'))
    args.append(Argument('dropout', [0.1]))#, add_to_name_as='dropout'))
    args.append(Argument('max_seq_len', [500]))#, add_to_name_as='max_seq_len'))
    args.append(Argument('seq_len', [7]))#, add_to_name_as='seq_len'))
    args.append(Argument('batch_size', [16]))#, add_to_name_as='batch_size'))
    args.append(Argument('batches_per_epoch', [0]))
    args.append(Argument('learning_rate', [1e-3]))
    args.append(Argument('lr_annealing', [0]))
    args.append(Argument('train_on_last_n_steps', [None]))# [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140], add_to_name_as='train_length'))

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    ubermain(args, n_runs, results_folder, experiment, data_dir, test_split_file, list_of_participants, n_cpu, n_proc_per_gpu)





