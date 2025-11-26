import os
import glob
import sys
sys.path.append('..')
import utils
import re

main_dir = '/home/janik.fechtelpeter/Documents/ordinal-bptt/results/dim_zy_MRT1_EveryValidDay_Smoothed'

all_hyper_paths = glob.glob(f'{main_dir}/**/*.pkl', recursive=True)
n = 0
for hyper_path in all_hyper_paths:
    path = os.path.split(hyper_path)[0]
    args = utils.load_args(path)
    participant_id_match = re.match('.*?_([0-9]+).csv', args['data_path'])
    if participant_id_match is not None:
        args['participant'] = participant_id_match.group(1)
        n += 1
    utils.save_args(args, path)

print(f'{n}/{len(all_hyper_paths)} hyperparameter files updated')