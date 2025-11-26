import os, sys
sys.path.append(os.getcwd())
sys.path.append('..')

import glob
import time
import concurrent.futures
import torch as tc

from recognition_models.rec_model_utils import estimate_observation_covariance_from_residuals
import data_utils
import argparse

if __name__ == '__main__':

    tc.set_num_threads(1)

    parser = argparse.ArgumentParser(description="Estimate observation covariance matrices in parallel.")
    parser.add_argument('--max_workers', '-w', type=int, default=10, help='Maximum number of parallel workers')
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
        'diagonal_covariance': True,
        'save_model': True,
        'filename': 'empirical_covariance.pt'
    }

    def estimate_covariance(dir_path_index):
        dir_path = model_dirs[dir_path_index]
        return dir_path, estimate_observation_covariance_from_residuals(dir_path, diagonal_cov=params['diagonal_covariance']).detach().clone()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(estimate_covariance, i) for i in range(len(model_dirs))]
        errors = 0
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                if params['save_model']:
                    tc.save(result[1], f'{result[0]}/{params["filename"]}')
                print(f"Job {i+1}/{len(model_dirs)}: Success")
            except Exception as e:
                print(f"Job {i+1}/{len(model_dirs)}: Error processing directory: {e}")
                errors += 1
        if errors > 0:
            print(f"Finished with {errors} errors.")