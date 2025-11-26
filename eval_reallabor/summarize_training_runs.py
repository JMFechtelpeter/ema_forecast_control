import sys, os
sys.path.append('..')
sys.path.append(os.getcwd())
import glob
import yaml
import pandas as pd
from eval_reallabor_utils import summarize_training_run
import utils

for main_dir in sorted(glob.glob(utils.join_base_path('bptt/results/*'))):

    if os.path.isdir(main_dir):
        summarize_training_run(main_dir, summary_file=utils.join_base_path('bptt/results/_training_summary.yml'))

with open('../results/_training_summary.yml', 'r') as file:
    summary = yaml.safe_load(file)
summary = pd.DataFrame.from_dict(summary)
summary.to_csv(utils.join_base_path('bptt/results/_training_summary.csv'))

# main_dir = '/home/janik.fechtelpeter/Documents/bptt/results/Reallabor1.0PremiumParticipants'
# summarize_training_run(main_dir, summary_file=utils.join_base_path('bptt/results/_training_summary.yml'))