import os
import glob
from tqdm import tqdm

prune_tb = True
prune_plots = False
prune_dir = 'results'

if prune_tb:
    print(f'Deleting tensorboard files from {prune_dir}')
    for file in tqdm(glob.glob(f'{prune_dir}/**/events.out*', recursive=True)):
        os.remove(file)
if prune_plots:
    print(f'Deleting plot files from {prune_dir}')
    for file in tqdm(glob.glob(f'{prune_dir}/**/*.png', recursive=True)):
        if not '00_summary' in file:
            os.remove(file)