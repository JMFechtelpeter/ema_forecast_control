import os
import pandas as pd

def determine_best_run(runs_path: str) -> str:
    losses = {}
    for model_dir in os.listdir(runs_path):
        try:
            loss = pd.read_csv(os.path.join(runs_path, model_dir,'loss.csv'))
            if (loss['validation_loss']==0).all():
                losses[1000] = model_dir
            else:
                losses[loss['validation_loss'].min()] = model_dir
        except:
            pass
    min_losses = min(losses.keys())
    best_run = losses[min_losses]
    return best_run