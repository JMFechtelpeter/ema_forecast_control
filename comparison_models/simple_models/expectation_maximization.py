# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 18:16:15 2023

@author: Elisa.Stegmeier
"""
from typing import Optional
import torch as tc
from torch._C import _LinAlgError
from tqdm import tqdm

def KalmanFilterSmoother(x: tc.Tensor, mu0: tc.Tensor, L0: tc.Tensor, 
                         A: tc.Tensor, C: tc.Tensor, Gamma: tc.Tensor, Sigma: tc.Tensor, 
                         u: Optional[tc.Tensor]=None, B: Optional[tc.Tensor]=None,
                         use_pinv_instead_of_leastsq: bool=True):
    
    p = Sigma.shape[0]                #dimensionality of latent states
    n = x.shape[1]                    #number of time steps
    q = x.shape[0]                    #number of observations
    h = mu0.shape[0]                  #number of hidden states
   
    #initialize variables for filter and smoother
    L = tc.zeros((p,p,n))           # measurement covariance matrix
    L[:,:,0] = L0                   # prior covariance
    mu_p  = tc.zeros((p,n))         # predicted expected value
    mu_p[:,0] = mu0                 # prior expected value
    mu = tc.zeros((p,n))            # filter expected value
    V = tc.zeros((p,p,n))           # filter covariance matrix
    K = tc.zeros((p,q,n))           # Kalman Gain
    Vhat  = tc.zeros((p,p,n))       # smoother covariance matrix
    muhat = tc.zeros((p,n))         # smoother expected value
    Vhat1 = tc.zeros((p,p,n))       # lag-1 covariance matrix (t,t-1)
    

    # KALMAN FILTER 
    # --------------------------------------------------------------------------
    # first step
    if use_pinv_instead_of_leastsq:
        K[:,:,0] = (L[:,:,0] @ C.T ) @ tc.linalg.pinv( C @ L[:,:,0] @ C.T + Gamma)
    else:
        K[:,:,0] = tc.linalg.lstsq((C @ L[:,:,0] @ C.T).T + Gamma.T, C @ L[:,:,0].T).solution.T
    # The above is replaced by lstsq for numerical stability, see https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
    mu[:,0] = mu_p[:,0] + K[:,:,0] @ (x[:,0] - C @ mu_p[:,0] )
    V[:,:,0] = (tc.eye(p) - K[:,:,0] @ C) @ L[:,:,0]
    
    for t in range(1,n):        
        L[:,:,t] = A @ V[:,:,t-1] @ A.T + Sigma
        if use_pinv_instead_of_leastsq:
            K[:,:,t] = L[:,:,t] @ C.T @ tc.linalg.pinv(C @ L[:,:,t] @ C.T + Gamma)          #Kalman gain^
        else:
            K[:,:,t] = tc.linalg.lstsq((C @ L[:,:,t] @ C.T).T + Gamma.T, C @ L[:,:,t].T).solution.T
        # The above is replaced by lstsq for numerical stability, see https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
        if u is not None:
            mu_p[:,t] = A @ mu[:,t-1] + B @ u[:,t]                                         #model prediction
        else:
            mu_p[:,t] = A @ mu[:,t-1]
        if not tc.isnan(x[:,t]).any():
            mu[:,t] = mu_p[:,t]  + K[:,:,t] @ (x[:,t] - C @ mu_p[:,t])                 #filtered state
        else:
            mu[:,t] = mu_p[:,t]                                                        #if observation is missing, do not filter the state 
        V[:,:,t] = (tc.eye(p) - K[:,:,t] @ C) @ L[:,:,t]                               #filtered covariance 
    
    
    # SMOOTHER
    # --------------------------------------------------------------------------
    muhat[:,n-1] = mu[:,-1]     # last expected value of smoother is equal to filter
    Vhat[:,:,n-1] = V[:,:,-1]   # last covariance of smoother is equal to filter
    
    for t in range(n-2,-1,-1): # go backwards
        if use_pinv_instead_of_leastsq:
            J = V[:,:,t] @ A.T @ tc.linalg.pinv(L[:,:,t])
        else:
            J = tc.linalg.lstsq(L[:,:,t].T, A @ V[:,:,t].T).solution.T
        # The above is replaced by lstsq for numerical stability, see https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
        Vhat[:,:,t] = V[:,:,t] + J @ (Vhat[:,:,t+1] - L[:,:,t]) @ J.T
        muhat[:,t] = mu[:,t] + J @ (muhat[:,t+1] - A @ mu[:,t])
        Vhat1[:,:,t+1] = Vhat[:,:,t+1] @ J.T 

    
    return muhat, Vhat, Vhat1



def ParamEstimLDS(x: tc.Tensor, u: Optional[tc.Tensor], Ez: tc.Tensor, Vhat: tc.Tensor, Vhat1: tc.Tensor, C: Optional[tc.Tensor]=None,
                  use_pinv_instead_of_leastsq: bool=True) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor, tc.Tensor|None, tc.Tensor, tc.Tensor, tc.Tensor]:
    '''
    implements parameter estimation for LDS
    z_1 ~ N(mu0,S)
    z_t = A z_t-1 + B u_t-1 + e_t , e_t ~ N(0,S)
    x_t = C z_t + nu_t , nu_t ~ N(0,G)
    
    REQUIRED INPUTS:
    x: observed variables (q x n, with q= number obs., n= number time steps)
    Ez: state estimates -> = muhat
    Vhat: estimated state covariance
    Vhat1: estimated lag-1 state covariance
    
    OPTIONAL INPUTS:
    C: observation parameter matrix (will be computed in here if omitted or set to []).
    
    OUTPUTS:
    mu0: state first moment initial conditions
    A: transition matrix
    C: observation matrix
    B: input matrix (if u is not None, otherwise None)
    S: process noise diagonal covariance matrix
    G: observation noise diagonal covariance matrix
    ELL: expected (complete data) log-likelihood
    '''

    [p,T] = Ez.shape
    q = x.shape[0]
    r = u.shape[0] if u is not None else 0
    
    # compute additional moments & expectancy matrices required for parameter estim:
    x2   = tc.zeros((q,q,T))  # x*x'
    Ezz  = tc.zeros((p,p,T))  # second moments
    Ezz_extended = tc.zeros((p+r,p+r,T))
    Ezz1_extended = tc.zeros((p,p+r,T-1))# lag-1 second moments
    Ezx  = tc.zeros((p,q,T))  # E[z]*x'    
    
    for t in range(0,T):        
        x2[:,:,t]   = x[:,t].reshape((q,1)) @ x[:,t].reshape((q,1)).T # need to reshape from (15,) to (15,1)
        Ezz[:,:,t]  = Vhat[:,:,t] + Ez[:,t].reshape((p,1)) @ Ez[:,t].reshape((p,1)).T
        Ezz_extended[:p,:p,t] = Ezz[:,:,t]
        if u is not None:
            Ezz_extended[:p,p:,t] = Ez[:,t:t+1] @ u[:,t:t+1].T
            Ezz_extended[p:,:p,t] = u[:,t:t+1] @ Ez[:,t:t+1].T
            Ezz_extended[p:,p:,t] = u[:,t:t+1] @ u[:,t:t+1].T
        Ezx[:,:,t]  = Ez[:,t].reshape((p,1)) @ x[:,t].reshape((q,1)).T
        if t<(T-1):
            Ezz1_extended[:,:p,t] = Vhat1[:,:,t+1] + Ez[:,t+1:t+2] @ Ez[:,t:t+1].T
            if u is not None:
                Ezz1_extended[:,p:,t] = Ez[:,t+1:t+2] @ u[:,t:t+1].T
    
    # compute all parameters:
    mu0 = Ez[:,0]
    
    if use_pinv_instead_of_leastsq:
        A_extended = tc.sum(Ezz1_extended, dim=2) @ tc.linalg.pinv(tc.sum(Ezz_extended[:,:,:-1],dim=2)) 
    else:
        A_extended = tc.linalg.lstsq(tc.sum(Ezz_extended[:,:,:-1], dim=2).T, tc.sum(Ezz1_extended, dim=2).T).solution.T
    # The above is replaced by lstsq for numerical stability, see https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
    
    S = ( (tc.sum(Ezz[:,:,1:], dim=2) 
           - A_extended @ (tc.sum(Ezz1_extended, dim=2)).T 
           - tc.sum(Ezz1_extended, dim=2) @ A_extended.T 
           + A_extended @ tc.sum(Ezz_extended[:,:,:-1], dim=2) @ A_extended.T)
        / (T))
    S = tc.diag(tc.diag(S))
    
    if C is None:
        if use_pinv_instead_of_leastsq:
            C = tc.nansum(Ezx,dim = 2).T @ tc.linalg.pinv(tc.sum(Ezz, dim = 2))    
        else:
            C = tc.linalg.lstsq(tc.sum(Ezz, dim=2).T, tc.nansum(Ezx, dim=2)).solution.T
        if C is None:
            raise EM_Error('For some reason, C is still None, even though it was explicitly calculated.')
        # The above is replaced by lstsq for numerical stability, see https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
    
    G = tc.nanmean(x2, dim=2) - C @ tc.nanmean(Ezx, dim=2) - tc.nanmean(Ezx, dim=2).T @ C.T + C @ tc.mean(Ezz, dim=2) @ C.T
    G = tc.diag(tc.diag(G))
    
    
    # compute expected log-likelihood:
    if use_pinv_instead_of_leastsq:
        SA = tc.linalg.pinv(S) @ A_extended
    else:
        SA = tc.linalg.lstsq(S, A_extended).solution
    # The above is replaced by lstsq for numerical stability, see https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
    ASA = A_extended.T @ SA
    # GC =  tc.linalg.inv(G) @ C
    GC = tc.linalg.solve(G, C)
    # The above is replaced by solve for numerical stability, see https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv
    CGC = C.T @ GC
    
    # ELL1 = -(T *tc.log(tc.linalg.det(S)) + T * tc.log(tc.linalg.det(G)) - 2 * tc.trace(SA.T @ Ezz1_extended[:,:,0]) 
    #         + tc.trace(ASA @ Ezz_extended[:,:,0]) + tc.trace( CGC @ Ezz[:,:,0]) + tc.trace(tc.linalg.inv(G) @ x2[:,:,0])
    #         - 2 * tc.trace(GC @ Ezx[:,:,0])) /2
    
    # for t in range(1,T-1):
    #     ELL1 = ELL1 - (tc.trace(tc.linalg.inv(S) @ Ezz[:,:,t]) - 2 * tc.trace(SA.T @ Ezz1_extended[:,:,t])
    #     + tc.trace( ASA @ Ezz_extended[:,:,t]) + tc.trace( CGC @ Ezz[:,:,t]) + tc.trace(tc.linalg.inv(G) @ x2[:,:,t])
    #     - 2 * tc.trace(GC @ Ezx[:,:,t] )) / 2
    
    # ELL1 = ELL1 - (tc.trace(tc.linalg.inv(S) @ Ezz [:,:,T-1])
    #     + tc.trace(CGC @ Ezz[:,:,T-1]) + tc.trace(tc.linalg.inv(G) @ x2[:,:,T-1])
    #     - 2 * tc.trace(GC @ Ezx[:,:,T-1])) /2
        
    ELL = -(T *tc.log(tc.linalg.det(S)) + T * tc.log(tc.linalg.det(G)) - 2 * tc.trace(SA.T @ Ezz1_extended[:,:,0]) 
            + tc.trace(ASA @ Ezz_extended[:,:,0]) + tc.trace( CGC @ Ezz[:,:,0]) + tc.trace(tc.linalg.solve(G, x2[:,:,0]))
            - 2 * tc.trace(GC @ Ezx[:,:,0])) /2
    
    for t in range(1,T-1):
        if not tc.isnan(x2[:,:,t]).any():
            ELL = ELL - (tc.trace(tc.linalg.solve(S, Ezz[:,:,t])) - 2 * tc.trace(SA.T @ Ezz1_extended[:,:,t])
            + tc.trace( ASA @ Ezz_extended[:,:,t]) + tc.trace( CGC @ Ezz[:,:,t]) + tc.trace(tc.linalg.solve(G, x2[:,:,t]))
            - 2 * tc.trace(GC @ Ezx[:,:,t] )) / 2
    
    if not tc.isnan(x2[:,:,T-1]).any():
        ELL = ELL - (tc.trace(tc.linalg.solve(S, Ezz [:,:,T-1]))
            + tc.trace(CGC @ Ezz[:,:,T-1]) + tc.trace(tc.linalg.solve(G, x2[:,:,T-1]))
            - 2 * tc.trace(GC @ Ezx[:,:,T-1])) /2
    
    # The above is replaced by solve for numerical stability, see https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv

    B = A_extended[:,p:] if u is not None else None
    A = A_extended[:,:p]

    return mu0, A, B, C, S, G, ELL


def EM_algorithm(x: tc.Tensor, u: Optional[tc.Tensor], 
                 latent_size: int, max_iter: int=10000, tol: float=1e-6, 
                 max_A_eigval: float=tc.inf, use_tqdm: bool=True, pbar_descr: str='', n_initializations: int=10,
                 use_pinv_instead_of_leastsq: bool=False):
    
    """
    Expectation Maximization for Linear Dynamical Systems (LDS).

    Model:
        z_1 ~ N(mu0, S)
        z_t = A z_{t-1} + B u_{t-1} + e_t,   e_t ~ N(0, S)
        x_t = C z_t + nu_t,                  nu_t ~ N(0, G)

    Args:
        x (torch.Tensor): Observed variables (shape: [q, n], q = number of observations, n = number of time steps).
        u (Optional[torch.Tensor]): Input variables (shape: [r, n], r = number of inputs, n = number of time steps).
        latent_size (int): Number of latent states (p).
        max_iter (int, optional): Maximum number of EM iterations. Default is 10000.
        tol (float, optional): Convergence tolerance. Default is 1e-6.
        max_A_eigval (float, optional): Maximum allowed eigenvalue of A. Default is torch.inf.
        use_tqdm (bool, optional): Whether to use tqdm progress bar. Default is True.
        pbar_descr (str, optional): Description for progress bar. Default is ''.
        n_initializations (int, optional): Number of random initializations to try. Default is 10.
        use_pinv_instead_of_leastsq (bool, optional): Use pinv instead of lstsq for numerical stability. Default is False.

    Returns:
        Tuple containing:
            - A (torch.Tensor): Transition matrix.
            - B (Optional[torch.Tensor]): Input matrix (None if u is None).
            - C (torch.Tensor): Observation matrix.
            - Gamma (torch.Tensor): Observation noise diagonal covariance matrix.
            - Sigma (torch.Tensor): Process noise diagonal covariance matrix.
            - mu0 (torch.Tensor): Initial state mean.
            - ELL (torch.Tensor): Expected (complete data) log-likelihood.

    Raises:
        EM_Error: If initialization fails to produce a valid log-likelihood.
    """

    ELL = tc.tensor([tc.nan])

    for init in range(n_initializations):

        observation_size = x.shape[0]
        A = tc.diag(tc.rand(latent_size))            
        C = tc.randn(observation_size, latent_size)    
        Gamma = tc.diag(tc.rand(observation_size))+0.001
        Sigma = tc.diag(tc.rand(latent_size))+0.001 # Sigma = L0????
        mu0 = tc.randn(latent_size)
        if u is None:
            B = None
        else:
            B = tc.rand(latent_size, u.shape[0])

        #try E-step and M-step once to check if the initialization works
        Ez, Vhat, Vhat1 = KalmanFilterSmoother(x, mu0, Sigma, A, C, Gamma, Sigma, u, B, use_pinv_instead_of_leastsq=use_pinv_instead_of_leastsq)
        try:
            _, _, _, _, _, _, ELL = ParamEstimLDS(x, u, Ez, Vhat, Vhat1, use_pinv_instead_of_leastsq=use_pinv_instead_of_leastsq)
        except:
            continue

        if tc.isnan(ELL):
            continue
        
        # EM loop
        ELL=tc.zeros(max_iter)
        LLR=tc.inf
        i=0
        if use_tqdm:
            iterator = tqdm(range(max_iter), desc=pbar_descr)
        else:
            iterator = range(max_iter)

        for i in iterator:
                
            # while i<max_iter and LLR>(tol*abs(ELL[0])):        
            # E-step: Kalman filter-smoother
            Ez, Vhat, Vhat1 = KalmanFilterSmoother(x, mu0, Sigma, A, C, Gamma, Sigma, u, B, use_pinv_instead_of_leastsq=use_pinv_instead_of_leastsq)
            # M-step: Parameter estimation
            try:
                mu0_ten, A_ten, B_ten, C_ten, Sigma_ten, Gamma_ten, ELL[i] = ParamEstimLDS(x, u, Ez, Vhat, Vhat1, use_pinv_instead_of_leastsq=use_pinv_instead_of_leastsq)    
            except RuntimeError:
                if isinstance(iterator, tqdm):
                    iterator.set_description(' - '.join((pbar_descr, f'init {init} EM Error: Ezz and/or Ezz1 contained inf values.')))
                    iterator.close()
                break
            except _LinAlgError:
                if isinstance(iterator, tqdm):
                    iterator.set_description(' - '.join((pbar_descr, f'init {init} EM Error: Gamma not invertible.')))
                    iterator.close()
                break
            # compute log-likelihood ratio
            if i>0:
                LLR=ELL[i]-ELL[i-1]
                if isinstance(iterator, tqdm):
                    iterator.set_description(' - '.join((pbar_descr, f'init {init} LLR={LLR:.4f}')))
            if not (ELL[i].isnan() or ELL[i].isinf()):
                mu0 = mu0_ten
                A = A_ten
                B = B_ten
                C = C_ten
                Sigma = Sigma_ten
                Gamma = Gamma_ten
                if LLR<(tol*abs(ELL[0])):
                    if tc.abs(tc.linalg.eig(A)[0]).max() < max_A_eigval:
                        ELL = ELL[:i] # cut off unnecessary zeros 
                        return A, B, C, Gamma, Sigma, mu0, ELL
            else:
                if isinstance(iterator, tqdm):
                    iterator.set_description(f'{pbar_descr} - init {init} EM Error: expected log likelihood is nan or inf.')
                    iterator.close()
                break
            
    raise EM_Error(' - '.join((pbar_descr, f'Initialization failed: there were no initial parameter values found ({n_initializations} tries) such that the expected log-likelihood is not NaN.')))


class EM_Error(Exception):

    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return f"EM Error: {self.args[0]}"
        
    


if __name__=='__main__':
    import matplotlib.pyplot as plt
    data = tc.tensor([[0.5085,0.6443,0.3507,0.6225,0.4709,0.2259,0.3111,0.9049,0.2581,0.6028],
                        [0.5108,0.3786,0.9390,0.5870,0.2305,0.1707,0.9234,0.9797,0.4087,0.7112],
                        [0.8176,0.8116,0.8759,0.2077,0.8443,0.2277,0.4302,0.4389,0.5949,0.2217],
                        [0.7948,0.5328,0.5502,0.3012,0.1948,0.4357,0.1848,0.1111,0.2622,0.1174]]).T
    inputs = None
    A, B, C, Gamma, Sigma, mu0, ELL = EM_algorithm(data.T, inputs, 3)
    plt.plot(ELL)
    plt.savefig('ELL.png')
