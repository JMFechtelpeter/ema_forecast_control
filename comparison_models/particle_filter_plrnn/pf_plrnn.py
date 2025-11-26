import os
import sys
from typing import Optional
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.special import logsumexp
import math

# print(os.getcwd())
# print(sys.path)
try:
    sys.path.append('../..')
    from dataset.multimodal_dataset import MultimodalDataset
except:
    sys.path.append(os.getcwd())
    from dataset.multimodal_dataset import MultimodalDataset
# from _main.dataset.multimodal_dataset import MultimodalDataset
import utils
import data_utils
sys.path.append(data_utils.join_ordinal_bptt_path('comparison_models/particle_filter_plrnn'))
import pf_utils

def _softplus_eps(x, eps=1e-6):
    return F.softplus(x) + eps

class PF_PLRNN(nn.Module):

    def __init__(self, args: Optional[dict] = None, dataset: Optional[MultimodalDataset] = None,
                 load_model_path: Optional[str] = None, resume_epoch: Optional[int] = None):
        if args is not None:
            self.args = args
            self.init_parameters()
        elif load_model_path is not None:
            self.init_from_model_path(load_model_path, resume_epoch)

    def init_from_model_path(self, load_model_path: str, resume_epoch: Optional[int]=None):
        
        self.args = utils.load_args(load_model_path)
        self.init_parameters()
        if resume_epoch is None:
            resume_epoch = utils.infer_latest_epoch(load_model_path)
        path = os.path.join(load_model_path, f'model_{resume_epoch}.pt')
        state_dict = tc.load(path)
        self.update_parameters(state_dict)

    def init_parameters(self):
        self.params = {}
        self.params['AW'] = self._init_uniform((self.args['dim_z'], self.args['dim_z']))
        self.params['C'] = self._init_uniform((self.args['dim_z'], self.args['dim_s']))
        self.params['B'] = self._init_uniform((self.args['dim_x'], self.args['dim_z']))
        self.params['Q'] = self._init_uniform((self.args['dim_z']), gain=0.01)
        self.params['R'] = self._init_uniform((self.args['dim_x']), gain=0.01)
        self.params['x_mean'] = nn.Parameter(tc.zeros(self.args['dim_x']), requires_grad=False)
        self.params['x_std']  = nn.Parameter(tc.ones(self.args['dim_x']), requires_grad=False)

    def _init_uniform(self, shape: int|tuple, gain: float=1.0) -> nn.Parameter:
        tensor = tc.empty(shape, dtype=tc.get_default_dtype())
        r = 1 / math.sqrt(shape[-1]) * gain if isinstance(shape, tuple) else 1.0 * gain
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)

    def update_parameters(self, new_parameters: dict):
        for k, v in new_parameters.items():
            self.params[k].data = v.data.clone()
    @property
    def d(self): return self.args['dim_z']
    @property
    def m(self): return self.args['dim_s']
    @property
    def p(self): return self.args['dim_x']

    def _split_A_W(self):
        """
        Returns:
          A_diag : (d,) vector with diagonal of A
          W_off  : (d,d) matrix with zero diagonal (off-diagonal W)
        """
        AW = self.params['AW']
        A_diag = tc.diag(AW)
        W_off = AW - tc.diag(A_diag)
        return A_diag, W_off

    def _Q_std(self):
        # elementwise stds for process noise (positive)
        return _softplus_eps(self.params['Q'])  # (d,)

    def _R_std(self):
        # elementwise stds for observation noise (positive)
        return _softplus_eps(self.params['R'])  # (p,)

    # --------------------------
    # Core model maps
    # --------------------------
    @tc.no_grad()
    def emission_mean(self, z: tc.Tensor) -> tc.Tensor:
        """
        x_mean = B z
        z: (..., d) -> x_mean: (..., p)
        """
        B = self.params['B']            # (p,d)
        return z @ B.T                  # (..., p)

    @tc.no_grad()
    def f_mean(self, z: tc.Tensor, u: tc.Tensor) -> tc.Tensor:
        """
        μ_{t+1} = A z + W ReLU(z) + C u   (bias-free version)
        z: (..., d), u: (..., m) -> (..., d)
        """
        A_diag, W_off = self._split_A_W()       # (d,), (d,d)
        Cz = F.relu(z) @ W_off.T                 # (..., d)
        Az = z * A_diag                          # (..., d)
        Cu = u @ self.params['C'].T              # (..., d)
        return Az + Cz + Cu                      # (..., d)

    def sample_z_next(self, z: tc.Tensor, u: tc.Tensor, rng: tc.Generator | None = None) -> tc.Tensor:
        """
        Draw z_{t+1} ~ N(f_mean(z,u), diag(q^2))
        """
        mu = self.f_mean(z, u)                      # (..., d)
        q = self._Q_std()                           # (d,)
        noise = tc.normal(mean=0.0, std=1.0, size=mu.shape, generator=rng, device=mu.device, dtype=mu.dtype)
        return mu + noise * q                      # (..., d)

    def log_p_x_given_z(self, x: tc.Tensor, z: tc.Tensor) -> tc.Tensor:
        """
        Log N(x; B z, R) with diagonal R.
        x: (..., p), z: (..., d) -> (...,) log-density per sample
        """
        x_mean = self.emission_mean(z)             # (..., p)
        r = self._R_std()                          # (p,)
        resid = x - x_mean                         # (..., p)
        # elementwise Gaussian logpdf with diag cov
        log_det = tc.log(r).sum() * 2.0            # log |R| = 2 * sum log r_i
        quad = (resid / r)**2
        # sum over last dim (p), keep batch dims
        ld = -0.5 * (quad.sum(dim=-1) + log_det + self.p * math.log(2.0 * math.pi))
        return ld                                   # (...,)

    def log_p_z_next_given_z(self, z_next: tc.Tensor, z: tc.Tensor, u: tc.Tensor) -> tc.Tensor:
        """
        Log N(z_next; f_mean(z,u), Q) with diagonal Q.
        z_next, z: (..., d); u: (..., m) -> (...,)
        """
        mu = self.f_mean(z, u)                     # (..., d)
        q = self._Q_std()                          # (d,)
        resid = z_next - mu                        # (..., d)
        log_det = tc.log(q).sum() * 2.0            # log |Q| = 2 * sum log q_i
        quad = (resid / q)**2
        ld = -0.5 * (quad.sum(dim=-1) + log_det + self.d * math.log(2.0 * math.pi))
        return ld                                   # (...,)

    # --------------------------
    # Initial state (standard normal by default)
    # --------------------------
    def sample_z0(self, n: int, rng: tc.Generator | None = None, device=None, dtype=None) -> tc.Tensor:
        """
        Sample z0 ~ N(0, I_d); shape (n, d)
        """
        device = device if device is not None else self.params['AW'].device
        dtype  = dtype  if dtype  is not None else self.params['AW'].dtype
        return tc.normal(mean=0.0, std=1.0, size=(n, self.d), generator=rng, device=device, dtype=dtype)

    def log_p_z0(self, z0: tc.Tensor) -> tc.Tensor:
        """
        Log N(z0; 0, I_d) per sample.
        """
        quad = (z0**2).sum(dim=-1)
        ld = -0.5 * (quad + self.d * math.log(2.0 * math.pi))
        return ld

    # --------------------------
    # Convenience: simulate + forecast
    # --------------------------
    @tc.no_grad()
    def simulate(self, T: int, u: tc.Tensor | None = None, n_sims: int = 1, rng: tc.Generator | None = None):
        """
        Simulate T time steps, starting from z0 ~ N(0,I).
        Returns:
          z: (T, n_sims, d), x: (T, n_sims, p)
        """
        device = self.params['AW'].device
        dtype  = self.params['AW'].dtype
        if u is None:
            u = tc.zeros((T, n_sims, self.m), device=device, dtype=dtype)
        elif u.dim() == 2:  # (T, m) -> (T, n_sims, m)
            u = u.unsqueeze(1).expand(T, n_sims, self.m)

        z = tc.empty((T, n_sims, self.d), device=device, dtype=dtype)
        x = tc.empty((T, n_sims, self.p), device=device, dtype=dtype)
        z_t = self.sample_z0(n_sims, rng=rng, device=device, dtype=dtype)
        for t in range(T):
            x[t] = self.emission_mean(z_t) + tc.normal(0.0, 1.0, size=(n_sims, self.p), generator=rng, device=device, dtype=dtype) * self._R_std()
            z[t] = z_t
            z_t = self.sample_z_next(z_t, u[t], rng=rng)
        return z, x

    @tc.no_grad()
    def forecast(self, zT: tc.Tensor, u_future: tc.Tensor, T: int, n_paths: int = 100, rng: tc.Generator | None = None):
        """
        Given a filtered state draw zT: (n_paths, d) and future inputs u_future: (T, m),
        simulate T-step predictive paths.
        Returns:
          z_f: (T, n_paths, d), x_f: (T, n_paths, p)
        """
        device, dtype = zT.device, zT.dtype
        if zT.dim() == 1:
            zT = zT.unsqueeze(0).expand(n_paths, self.d)
        z_f = tc.empty((T, n_paths, self.d), device=device, dtype=dtype)
        x_f = tc.empty((T, n_paths, self.p), device=device, dtype=dtype)
        z_t = zT
        for h in range(T):
            z_t = self.sample_z_next(z_t, u_future[h].expand(n_paths, self.m), rng=rng)
            z_f[h] = z_t
            x_f[h] = self.emission_mean(z_t) + tc.normal(0.0, 1.0, size=(n_paths, self.p), generator=rng, device=device, dtype=dtype) * self._R_std()                    
        return z_f, x_f
    
    @tc.no_grad()
    def forecast_pf(
        self,
        x_hist: Tensor,          # (T_hist, p) in ORIGINAL scale
        u_hist: Tensor,          # (T_hist, m) standardized already
        x_test: Tensor,          # (H, p) in ORIGINAL scale (to score)
        u_test: Tensor,          # (H, m) standardized already
        n_particles: int = 1000,
        gen: tc.Generator | None = None,
        return_paths: bool = False,
    ):
        """
        1) Filters on (x_hist, u_hist) with APF to approximate p(z_T | data).
        2) Samples particles from that posterior.
        3) Propagates H steps with u_test (no resampling), draws x_t.
        4) Returns de-standardized forecast mean and MSE vs x_test (original domain).
        """
        assert 'x_mean' in self.params and 'x_std' in self.params, \
            "Model must carry x_mean/x_std from training to forecast in original domain."
        device, dtype = x_hist.device, x_hist.dtype
        p = self.p

        # standardize history x
        xh = (x_hist - self.params['x_mean']) / self.params['x_std']

        # ---- filter history with APF to get z_T particles + weights
        z_T, w_T, loglik = self._apf_filter_final(xh, u_hist, n_particles, gen)

        # ---- sample initial z_T for forecasting
        idx = tc.multinomial(w_T, num_samples=n_particles, replacement=True, generator=gen)
        z_t = z_T[idx]  # (N, d)

        H = x_test.shape[0]
        z_f = tc.empty(H, n_particles, self.d, device=device, dtype=dtype)
        x_f = tc.empty(H, n_particles, self.p, device=device, dtype=dtype)

        r_std = F.softplus(self.params['R']) + 1e-6  # (p,)

        for h in range(H):
            # propagate (no weights, no resampling beyond T)
            z_t = self.sample_z_next(z_t, u_test[h], rng=gen)
            # emit
            x_mean = self.emission_mean(z_t)                       # (N,p)
            x_draw = x_mean + tc.randn(x_mean.shape, generator=gen, device=x_mean.device, dtype=x_mean.dtype) * r_std
            x_f[h] = x_draw
            z_f[h] = z_t
            
        # de-standardize forecasted x
        x_fore = x_f * self.params['x_std'] + self.params['x_mean']    # (H, N, p)
        x_fore_mean = x_fore.mean(dim=1)           # (H, p)

        mse = float(((x_fore_mean - x_test)**2).nanmean().item())

        out = {
            "x_fore_mean": x_fore_mean,
            "mse": mse,
            "loglik_hist_std": float(loglik),  # PF loglik on standardized history
        }
        if return_paths:
            out["x_fore_paths"] = x_fore        # (H, N, p)
            out["z_fore_paths"] = z_f           # (H, N, d)
        return out
    
    # --- APF filter final (history) ---
    @tc.no_grad()
    def _apf_filter_final(self, x: tc.Tensor, u: tc.Tensor, N: int, gen: tc.Generator | None):
        """
        APF over (x,u) to final time; (masked) returns (z_T, w_T, loglik) on standardized x.
        """
        T, p = x.shape
        device, dtype = x.device, x.dtype

        z_t = self.sample_z0(N, rng=gen, device=device, dtype=dtype)
        r_std = F.softplus(self.params['R']) + 1e-6

        x0_mean = self.emission_mean(z_t)
        logw = pf_utils.masked_log_diag_gauss(x[0].expand_as(x0_mean), x0_mean, r_std)
        logZ = logsumexp(logw, 0) 
        w = tc.exp(logw - logZ)
        loglik = float((logZ - math.log(N)).item())

        for t in range(T - 1):
            mu_next = self.f_mean(z_t, u[t])
            x_pred  = self.emission_mean(mu_next)
            log_m   = pf_utils.masked_log_diag_gauss(x[t+1].expand_as(x_pred), x_pred, r_std)
            eps = 1e-32
            log_w = tc.log(w.clamp_min(eps))                      # w from previous step
            log_r = log_w + log_m
            log_r_norm = log_r - logsumexp(log_r, 0)
            r = tc.exp(log_r_norm)

            if not tc.isfinite(r).all() or float(r.sum().item()) == 0.0:
                r = w / w.sum()

            idx = pf_utils.systematic_resample(r, gen)
            anc = z_t[idx]
            m_ai_log = log_m[idx]

            z_prop = self.sample_z_next(anc, u[t], rng=gen)

            xnext_mean = self.emission_mean(z_prop)
            log_num = pf_utils.masked_log_diag_gauss(x[t+1].expand_as(xnext_mean), xnext_mean, r_std)
            log_wcorr = log_num - m_ai_log
            logZ2 = logsumexp(log_wcorr, 0)
            w = tc.exp(log_wcorr - logZ2)

            loglik += float(logsumexp(log_w + log_m, 0).item() + (logZ2 - math.log(N)))
            z_t = z_prop

        return z_t, w, loglik
