import os
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append('..')
import numpy as np
import torch as tc
import pandas as pd
from tqdm import tqdm
import eval_reallabor_utils

def lyapunov_spectrum(model, T, T_trans=100, ons=5):

    # evolve for transient time Tₜᵣ
    data = model.dataset.data()
    tmp, latent = model.generate_free_trajectory(*data, T_trans)
    # initialize
    z=latent[-1]
    y = np.zeros(z.shape[0])
    # initialize as Identity matrix
    Q = np.eye((z.shape[-1]))   

    for t in range(T):        
        z=model.latent_model.generate_step(z)
        state = (z)
        jacobians=tc.autograd.functional.jacobian(model.latent_model.generate_step, state)[0]
        # compute jacobian
        Q = jacobians * Q
        if (t%ons == 0):
            # reorthogonalize
            Q, R = np.linalg.qr(Q)
            # accumulate lyapunov exponents
            y += np.log(np.abs(np.diag(R)))
    return y / T


def calculate_lyapunov_spectra(main_dir):

    model_dirs = eval_reallabor_utils.get_model_folders(main_dir)
    model_dirs = model_dirs
    spectra = []
    for md in tqdm(model_dirs):
        try:
            model, _ = eval_reallabor_utils.load_model_and_data(md, allow_test_inputs=True)
        except Exception as e:
            print(e)
            continue
        spectra.append(np.sort(lyapunov_spectrum(model, 200, 10))[::-1])
    spectra = pd.DataFrame(index=model_dirs, data=np.array(spectra))
    return spectra
        
    
if __name__=='__main__':
    
    main_dir = '/home/janik.fechtelpeter/Documents/bptt/results/Reallabor1.0BestDaysNoInputsTrainLength2'
    spectra = calculate_lyapunov_spectra(main_dir)
    spectra.to_csv('lyapunov.csv')