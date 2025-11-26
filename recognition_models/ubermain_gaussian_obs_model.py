import os, sys
sys.path.append(os.getcwd())
sys.path.append('..')

import glob
import time
import concurrent.futures
import torch as tc

from recognition_models.train_gaussian_obs_model import train_gaussian_obs_model
import data_utils
import argparse

if __name__ == '__main__':

    tc.set_num_threads(1)

    parser = argparse.ArgumentParser(description="Train Gaussian observation models in parallel.")
    parser.add_argument('--max_workers', '-w', type=int, default=1, help='Maximum number of parallel workers')
    args = parser.parse_args()

    model_folders = [
                    #  data_utils.join_base_path('ordinal-bptt/results/v3_MRT2_every_day'),
                    #  data_utils.join_base_path('ordinal-bptt/results/v3_MRT3_every_day'),
                    #  data_utils.join_base_path('ordinal-bptt/results/v2_MRT2_every_day_x6'),
                     data_utils.join_base_path('ordinal-bptt/results/v2_MRT3_every_day_x6')
                    ]

    model_dirs = [os.path.split(p)[0] for k in range(len(model_folders)) for p in glob.glob(f'{model_folders[k]}/*/*/*.pt')]
    model_dirs = sorted(set(model_dirs))    

    print(f"Found {len(model_dirs)} model directories to process. Proceed in 10 seconds.")
    time.sleep(10)

    params = {
        'lr': 0.01,
        'n_epochs': 1000,
        'batch_size': 256,
        'l2_reg_gamma': 0,
        'penalty_on_diff_to_B': 200,
        'penalty_on_diff_to_B_inv': 0,
        'create_plots': False,
        'save_model': True,
        'filename': 'gaussian_obs_model_B_penalty.pt'
    }

    def train_on_dir(i):
        dir_path = model_dirs[i]
        params['pbar_desc'] = f'Job {i+1}/{len(model_dirs)}'
        tc.save(params, os.path.join(dir_path, params['filename']+'params.pt'))
        return train_gaussian_obs_model(dir_path, **params)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(train_on_dir, i) for i in range(len(model_dirs))]
        errors = 0
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
            except Exception as e:
                print(f"Job {i+1}/{len(model_dirs)}: Error processing directory: {e}")
                errors += 1
        if errors > 0:
            print(f"Finished with {errors} errors.")