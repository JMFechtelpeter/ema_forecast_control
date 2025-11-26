import os
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from dataclasses import dataclass
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import pf_utils

# ===============================================================
# Base Estimator class
# ===============================================================

class BaseEstimator:
    """Common interface for all estimators."""
    def __init__(self, model):
        self.model = model
        self.theta = {k: v.clone().detach().requires_grad_(True)
                      for k, v in model.params.items()}
        self.epoch = None

    def update_model(self):
        """Write local theta back into model.params"""
        for k, v in self.theta.items():
            self.model.params[k].data = v.data.clone()

    def fit(self, x, u):
        raise NotImplementedError

    def save(self, model_path: str):
        if self.epoch is not None:
            tc.save(self.model.params, os.path.join(model_path, f'model_{self.epoch}.pt'))
        else:
            tc.save(self.model.params, os.path.join(model_path, 'untrained_model.pt'))

# ===============================================================
# MAP Estimator (EKF/UKF-based differentiable likelihood)
# ===============================================================

class MapEstimator(BaseEstimator):
    """
    Regularized MAP estimator using differentiable EKF-like likelihood.
    This is a placeholder; EKF/UKF forward pass should be implemented later.
    """

    def __init__(self, model, lr=1e-2, weight_decay=1e-3, max_iters=200):
        super().__init__(model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_iters = max_iters


    def forward_filter_loglik(self, x, u):
        """
        Placeholder for differentiable filter (EKF/UKF).
        Should return an approximate differentiable log-likelihood.
        For now, we'll use a crude Gaussian reconstruction loss
        to illustrate optimization structure.
        """
        # reconstruct via emission mean
        x_hat = self.model.emission_mean(x @ 0 + 0)  # dummy for shape
        # Replace with filter’s predictions
        loss = ((x - x_hat)**2).mean()
        return -loss  # interpret as pseudo-log-likelihood

    def log_prior(self):
        """Simple L2 regularization as log prior."""
        lp = 0.0
        for name, p in self.theta.items():
            if name in ['AW', 'C', 'B']:
                lp -= self.weight_decay * (p**2).sum()
        return lp

    def fit(self, x, u):
        opt = tc.optim.Adam(self.theta.values(), lr=self.lr)
        for it in range(self.max_iters):
            opt.zero_grad()
            loglik = self.forward_filter_loglik(x, u)
            logpost = loglik + self.log_prior()
            loss = -logpost  # maximize logpost
            loss.backward()
            opt.step()
            if it % 10 == 0:
                print(f"[MAP] iter {it:03d} loss={loss.item():.4f}")
        self.update_model()
        return self.model


# ===============================================================
# Particle EM Estimator
# ===============================================================

def logsumexp(logw: Tensor, dim=-1):
    m = logw.max(dim=dim, keepdim=True).values
    return m.squeeze(dim) + tc.log(tc.exp(logw - m).sum(dim=dim))

@dataclass
class EStepStats:
    S_zz: Tensor; S_xz: Tensor; S_xx: Tensor
    S_phiphi: list; S_zplus_phi: list
    S_zplus_sq: Tensor; S_mu_sq: Tensor; S_zplus_mu: Tensor
    Ez_T: Tensor; loglik: float; varloglik: float; mse: float
    T_dyn: int

class ParticleEMEstimator(BaseEstimator):
    """
    Particle EM using:
      • Auxiliary PF (APF) + systematic resampling (ESS < N/2)
      • FFBSi smoother with M trajectories
      • Diagonal Q,R; ridge regularization optional
    """
    def __init__(
        self,
        model,
        n_iter: int = 20,
        n_particles: int = 1000,
        n_smooth_paths: int = 20,
        ess_threshold: float = 0.5,
        ridge_B: float = 0.0,
        ridge_dyn: float = 0.0,
        noise_floor: float = 1e-4,
        val_split: float = 0.0,
        var_runs: int = 3,
        gen: tc.Generator | None = None,
        pbar_descr: str = "PEM (APF+FFBSi)",
        **kwargs
    ):
        super().__init__(model)
        self.n_iter = n_iter
        self.N = n_particles
        self.M = n_smooth_paths
        self.ess_thr = ess_threshold
        self.lam_B = ridge_B
        self.lam_dyn = ridge_dyn
        self.noise_floor = noise_floor
        self.val_split = val_split
        self.var_runs = var_runs
        self.gen = gen
        self.pbar_descr = pbar_descr

    def fit(self, x: Tensor, u: Tensor):
        # standardize x, keep stats on model for forecasting
        x_mean = x.nanmean(dim=0)
        x_std = ((((x - x.nanmean(dim=0, keepdim=True))**2).nanmean(dim=0))**0.5).clamp_min(1e-8)
        x_s = (x - x_mean) / x_std
        self.model.params['x_mean'] = x_mean.detach().clone()
        self.model.params['x_std']  = x_std.detach().clone()
        self.stats_history = []

        T = x.shape[0]
        T_val = int(T * self.val_split)
        if T_val > 0:
            x_tr, u_tr = x_s[:-T_val], u[:-T_val]
            x_val, u_val = x_s[-T_val:], u[-T_val:]
        else:
            x_tr, u_tr = x_s, u
            x_val = u_val = None

        pbar = trange(self.n_iter, desc=self.pbar_descr, leave=True)
        best_snapshot, best_val, best_epoch = None, float("inf"), None

        for it in pbar:
            stats = self.e_step(x_tr, u_tr)   # ---- E-step (APF + FFBSi)
            self.m_step(stats)                # ---- M-step
            self.stats_history.append(pd.Series({'Ez_T': stats.Ez_T.detach().numpy(),
                                                  'loglik': stats.loglik,
                                                  'varloglik': stats.varloglik,
                                                  'mse': stats.mse}))

            if x_val is not None:
                ll_val, _ = self._apf_loglik(x_val, u_val, self.N)
                nll_val = -ll_val / len(x_val)
                if nll_val < best_val - 1e-4:
                    best_val = nll_val
                    best_snapshot = {k: v.detach().clone() for k,v in self.model.params.items()}
                    best_epoch = it + 1
            else:
                nll_val = float('nan')

            pbar.set_description_str(
                f"{self.pbar_descr} | "
                f"PEM it={it+1}/{self.n_iter} | loglik={stats.loglik:.2f} "
                f"| varloglik={stats.varloglik:.3f} | MSE={stats.mse:.4f} "
                + (f"| valNLL={nll_val:.4f}" if x_val is not None else "")
            )

        if best_snapshot is not None:
            for k,v in best_snapshot.items():
                self.model.params[k].data = v.data.clone()
                self.epoch = best_epoch
        else:
            self.update_model()
            self.epoch = it + 1

        self.last_Ez_T = stats.Ez_T.detach().clone()
        self.stats_history = pd.concat(self.stats_history, axis=1).T.reset_index(drop=True)
        
        return self.model, self.stats_history

    # ---------------------------
    # E-step: APF + FFBSi
    # ---------------------------
    def e_step(self, x: tc.Tensor, u: tc.Tensor) -> EStepStats:
        T, p = x.shape
        T_dyn = T - 1
        d, m = self.model.d, self.model.m
        device, dtype = x.device, x.dtype

        # forward APF with storage
        zs, ws, znexts, loglik = self._apf_forward_store(x, u, self.N)

        # var of loglik
        varloglik = 0.0
        if self.var_runs > 1:
            ll_runs = [self._apf_loglik(x, u, self.N)[0] for _ in range(self.var_runs)]
            varloglik = float(np.var(ll_runs, ddof=1) if len(ll_runs) > 1 else 0.0)

        # --- accumulators (time-sums)
        # emission (per-output, missing-safe)
        S_xz = tc.zeros(p, d, device=device, dtype=dtype)              # row j: sum_t x_{tj} E[z_t]^T over observed j
        S_xx_diag = tc.zeros(p, device=device, dtype=dtype)            # sum_t x_{tj}^2 over observed j
        S_zz_by_j = [tc.zeros(d, d, device=device, dtype=dtype) for _ in range(p)]  # sum_t E[z_t z_t^T] over observed j
        T_obs_j = tc.zeros(p, device=device, dtype=dtype)              # counts per j

        # also keep a global S_zz if you want (not needed for emission with missing)
        S_zz_all = tc.zeros(d, d, device=device, dtype=dtype)

        # dynamics
        S_phiphi = [tc.zeros(1 + (d-1) + m + 1, 1 + (d-1) + m + 1, device=device, dtype=dtype) for _ in range(d)]
        S_zplus_phi = [tc.zeros(1, 1 + (d-1) + m + 1, device=device, dtype=dtype) for _ in range(d)]
        S_zplus_sq = tc.zeros(d, device=device, dtype=dtype)

        Ez_T_sum = tc.zeros(d, device=device, dtype=dtype)

        # smoothed paths loop
        for _ in range(self.M):
            z_path = self._ffbsi_sample_path(zs, ws, znexts, u)   # (T,d)
            Ez_t = z_path                                     # align with x_t at times 1..T
            Ezz_t = tc.einsum('ti,tj->tij', Ez_t, Ez_t)           # (T,d,d)
            S_zz_all += Ezz_t.sum(dim=0)

            # emission per j, only at observed times
            for j in range(p):
                obs_t = ~tc.isnan(x[:, j])                        # (T,)
                if obs_t.any():
                    S_xz[j]        += (x[obs_t, j][:, None] * Ez_t[obs_t]).sum(dim=0)
                    S_xx_diag[j]   += (x[obs_t, j] ** 2).sum()
                    S_zz_by_j[j]   += Ezz_t[obs_t].sum(dim=0)
                    T_obs_j[j]     += obs_t.sum()

            # dynamics stats
            z_prev = z_path[:-1]                                 # (T-1, d)
            z_next = z_path[1:]                                  # (T-1, d)
            relu_prev = F.relu(z_prev)

            for i in range(d):
                relu_mask = [k for k in range(d) if k != i]
                phi_i = tc.cat([
                    z_prev[:, i:i+1],               # (T-1, 1)
                    relu_prev[:, relu_mask],        # (T-1, d-1)
                    u[:-1],                         # (T-1, m)  align with z_prev at t=0..T-2
                    tc.ones(T-1, 1, device=device, dtype=dtype)
                ], dim=1)                           # (T-1, k_i)
                S_phiphi[i]    += phi_i.T @ phi_i
                S_zplus_phi[i] += z_next[:, i:i+1].T @ phi_i

            S_zplus_sq += (z_next**2).sum(dim=0)
            Ez_T_sum   += z_path[-1]

        # average over smoothed paths so stats are unbiased time-sums
        M = float(self.M)
        S_zz_all   /= M
        S_xz       /= M
        S_xx_diag  /= M
        for j in range(p):
            S_zz_by_j[j] /= M
        for i in range(d):
            S_phiphi[i]    /= M
            S_zplus_phi[i] /= M
        S_zplus_sq /= M
        Ez_T = Ez_T_sum / M

        # quick MSE diagnostic on observed entries only
        with tc.no_grad():
            # use a small avg over a few paths for Ez_t
            Ez_t_check = (tc.stack([self._ffbsi_sample_path(zs, ws, znexts, u) for _ in range(3)], dim=0).mean(dim=0))
            x_hat = self.model.emission_mean(Ez_t_check)  # (T,p) in standardized space
            obs_mask = ~tc.isnan(x)
            mse = float((((x_hat - x).where(obs_mask, tc.zeros_like(x))).pow(2).sum()
                         / obs_mask.sum().clamp_min(1)).item())

        # Package stats
        stats = EStepStats(
            S_zz=S_zz_all, S_xz=S_xz, S_xx=S_xx_diag,            # note S_xx is diag vector now
            S_phiphi=S_phiphi, S_zplus_phi=S_zplus_phi,
            S_zplus_sq=S_zplus_sq, S_mu_sq=tc.tensor(0.0), S_zplus_mu=tc.tensor(0.0),  # not needed with variance formula used
            Ez_T=Ez_T, loglik=float(loglik), varloglik=float(varloglik), mse=mse,
            T_dyn=T_dyn
        )
        # stash per-output Z sums & counts for M-step
        stats.S_zz_by_j = S_zz_by_j   # attach dynamically
        stats.T_obs_j   = T_obs_j
        return stats

    # ---------------------------
    # M-step (unchanged in spirit; ridge + constraints)
    # ---------------------------
    def m_step(self, stats: EStepStats):
        d, m, p = self.model.d, self.model.m, self.model.p
        device = self.model.params['AW'].device
        dtype  = self.model.params['AW'].dtype

        # ----- Emission: per-output ridge regression on observed times
        B = tc.zeros(p, d, device=device, dtype=dtype)
        R_std = tc.zeros(p, device=device, dtype=dtype)
        I_d = tc.eye(d, device=device, dtype=dtype)

        for j in range(p):
            Tj = int(stats.T_obs_j[j].item())
            if Tj == 0:
                # nothing observed for this dimension; keep previous row
                B[j] = self.theta['B'][j].data
                R_std[j] = F.softplus(self.theta['R'][j].data) + 1e-6
                continue
            Szz_j = stats.S_zz_by_j[j] + self.lam_B * I_d
            bx_j  = stats.S_xz[j] @ tc.linalg.solve(Szz_j, I_d)          # (d,)
            B[j] = bx_j
            # residual variance using sums
            # r_j^2 = (S_xx_j - 2 B_j S_xz_j^T + B_j S_zz_j_raw B_j^T) / Tj
            raw_Szz_j = stats.S_zz_by_j[j]  # without ridge
            num = stats.S_xx[j] - 2.0 * (B[j] @ stats.S_xz[j]) + (B[j] @ raw_Szz_j @ B[j])
            R_std[j] = num.clamp_min(self.noise_floor).div(float(Tj)).sqrt()

        self.theta['B'].data = B
        self.theta['R'].data = tc.log(tc.exp(R_std) - 1.0).clamp_min(math.log(self.noise_floor))

        # ----- Dynamics (unchanged)
        A_diag = tc.zeros(d, device=device, dtype=dtype)
        W_off  = tc.zeros(d, d, device=device, dtype=dtype)
        C_mat  = tc.zeros(d, m, device=device, dtype=dtype)
        b_vec  = tc.zeros(d, device=device, dtype=dtype)

        for i in range(d):
            k_i = 1 + (d - 1) + m + 1
            I_k = tc.eye(k_i, device=device, dtype=dtype)
            Sphiphi_reg = stats.S_phiphi[i] + self.lam_dyn * I_k
            m_i = stats.S_zplus_phi[i] @ tc.linalg.solve(Sphiphi_reg, I_k)  # (1,k_i)

            A_diag[i] = self._bound_Aii(m_i[0,0])
            relu_mask = [j for j in range(d) if j != i]
            W_off[i, relu_mask] = m_i[0, 1:1+(d-1)]
            C_mat[i, :] = m_i[0, 1+(d-1):1+(d-1)+m]
            b_vec[i]    = m_i[0, -1]

        W_off = W_off - tc.diag(tc.diag(W_off))

        # Robust per-dim formula (time-sum level):
        T_dyn = stats.T_dyn
        Q_var = tc.empty(d, device=device, dtype=dtype)
        for i in range(d):
            k_i = stats.S_phiphi[i].shape[0]
            I_k = tc.eye(k_i, device=device, dtype=dtype)
            m_i = stats.S_zplus_phi[i] @ tc.linalg.solve(stats.S_phiphi[i] + self.lam_dyn * I_k, I_k)
            quad  = (m_i @ stats.S_phiphi[i] @ m_i.T).squeeze()
            cross = (m_i @ stats.S_zplus_phi[i].T).squeeze()
            Q_var[i] = (stats.S_zplus_sq[i] - 2 * cross + quad) / T_dyn

        Q_std = Q_var.clamp_min(self.noise_floor).sqrt()

        AW = W_off + tc.diag(A_diag)
        self.theta['AW'].data = AW
        self.theta['C'].data  = C_mat
        self.theta['Q'].data  = tc.log(tc.exp(Q_std) - 1.0).clamp_min(math.log(self.noise_floor))

    def _bound_Aii(self, a_raw: Tensor, mu: float = 0.95, span: float = 0.10):
        return mu + span * tc.tanh(a_raw)

    # ---------------------------
    # Auxiliary PF (APF) internals
    # ---------------------------
    def _apf_forward_store(self, x: tc.Tensor, u: tc.Tensor, N: int):
        T, p = x.shape
        d = self.model.d
        device, dtype = x.device, x.dtype
        gen = self.gen

        # t=0
        z_t = self.model.sample_z0(N, rng=gen, device=device, dtype=dtype)
        r_std = F.softplus(self.model.params['R']) + 1e-6
        # weight on x_0 with mask
        x0 = x[0].clone()
        x0_mean = self.model.emission_mean(z_t)                 # (N,p)
        logw = pf_utils.masked_log_diag_gauss(x0.expand_as(x0_mean), x0_mean, r_std)  # <-- NEW
        logZ = logsumexp(logw)
        w_t = tc.exp(logw - logZ)
        loglik = float((logZ - math.log(N)).item())

        zs = [z_t.clone()]; ws = [w_t.clone()]; znexts = []

        for t in range(T - 1):
            # look-ahead
            mu_next = self.model.f_mean(z_t, u[t])              # (N,d)
            x_pred  = self.model.emission_mean(mu_next)         # (N,p)
            log_m   = pf_utils.masked_log_diag_gauss(x[t+1].expand_as(x_pred), x_pred, r_std)  # <-- NEW
            m_i = tc.exp(log_m - log_m.max())

            r = w_t * m_i
            r_sum = r.sum(); r = r / r_sum.clamp_min(1e-32)
            idx = pf_utils.systematic_resample(r, gen)
            anc = z_t[idx]; m_ai = m_i[idx].clamp_min(1e-32)

            z_prop = self.model.sample_z_next(anc, u[t], rng=gen)
            # correction
            xnext_mean = self.model.emission_mean(z_prop)
            log_num = pf_utils.masked_log_diag_gauss(x[t+1].expand_as(xnext_mean), xnext_mean, r_std)  # <-- NEW
            log_den = tc.log(m_ai)
            log_wcorr = log_num - log_den
            logZ2 = logsumexp(log_wcorr)
            w_next = tc.exp(log_wcorr - logZ2)

            loglik += float(tc.log(r_sum + 1e-32).item() + (logZ2 - math.log(N)))

            znexts.append(z_prop.clone()); zs.append(z_prop.clone()); ws.append(w_next.clone())

            ess_next = 1.0 / (w_next**2).sum()
            if ess_next < self.ess_thr * N:
                idx2 = pf_utils.systematic_resample(w_next, gen)
                z_t = z_prop[idx2]
                w_t = tc.full_like(w_next, 1.0 / N)
            else:
                z_t = z_prop; w_t = w_next

        return zs, ws, znexts, loglik

    def _apf_loglik(self, x: tc.Tensor, u: tc.Tensor, N: int):
        T, p = x.shape
        device, dtype = x.device, x.dtype
        gen = self.gen

        z_t = self.model.sample_z0(N, rng=gen, device=device, dtype=dtype)
        r_std = F.softplus(self.model.params['R']) + 1e-6

        x0_mean = self.model.emission_mean(z_t)
        logw = pf_utils.masked_log_diag_gauss(x[0].expand_as(x0_mean), x0_mean, r_std)
        logZ = logsumexp(logw); w = tc.exp(logw - logZ)
        loglik = float((logZ - math.log(N)).item())

        for t in range(T - 1):
            mu_next = self.model.f_mean(z_t, u[t])
            x_pred  = self.model.emission_mean(mu_next)
            log_m   = pf_utils.masked_log_diag_gauss(x[t+1].expand_as(x_pred), x_pred, r_std)
            m_i = tc.exp(log_m - log_m.max())
            r = w * m_i
            r_sum = r.sum(); r = r / r_sum.clamp_min(1e-32)

            idx = pf_utils.systematic_resample(r, gen)
            anc = z_t[idx]; m_ai = m_i[idx].clamp_min(1e-32)

            z_prop = self.model.sample_z_next(anc, u[t], rng=gen)
            xnext_mean = self.model.emission_mean(z_prop)
            log_num = pf_utils.masked_log_diag_gauss(x[t+1].expand_as(xnext_mean), xnext_mean, r_std)
            log_den = tc.log(m_ai)
            log_wcorr = log_num - log_den
            logZ2 = logsumexp(log_wcorr)
            w = tc.exp(log_wcorr - logZ2)

            loglik += float(tc.log(r_sum + 1e-32).item() + (logZ2 - math.log(N)))
            z_t = z_prop

        return loglik, w

    # ---------------------------
    # FFBSi (unchanged)
    # ---------------------------
    def _ffbsi_sample_path(self, zs, ws, znexts, u: Tensor) -> Tensor:
        T = len(zs) - 1
        N = zs[0].shape[0]
        gen = self.gen

        i_t = tc.multinomial(ws[T], num_samples=1, generator=gen).item()
        path = [None] * (T + 1)
        path[T] = zs[T][i_t]

        for t in range(T - 1, -1, -1):
            z_cloud = zs[t]
            w_t = ws[t]
            z_next_sel = path[t+1].expand(N, -1)
            log_trans = self.model.log_p_z_next_given_z(z_next_sel, z_cloud, u[t])  # (N,)
            log_prob = tc.log(w_t + 1e-32) + log_trans
            log_prob = log_prob - logsumexp(log_prob)
            prob = tc.exp(log_prob).clamp_min(0)
            prob = prob / prob.sum()
            i_prev = tc.multinomial(prob, num_samples=1, generator=gen).item()
            path[t] = z_cloud[i_prev]

        return tc.stack(path, dim=0)

    def plot_loss(self):
        fig, axes = plt.subplots(3,1, figsize=(5,9), sharex=True)
        axes[0].plot(self.stats_history['loglik'])
        axes[0].set(ylabel="Log Likelihood")
        axes[1].plot(self.stats_history['varloglik'])
        axes[1].set(ylabel="Var Log Likelihood")
        axes[2].plot(self.stats_history['mse'])
        axes[2].set(ylabel="MSE", xlabel='Iteration')
        return plt.gcf()


# ===============================================================
# Hybrid Estimator: MAP -> PEM
# ===============================================================

class HybridEstimator:
    """Runs MAP first, then PEM."""
    def __init__(self, map_cfg=None, pem_cfg=None):
        self.map_cfg = map_cfg or {}
        self.pem_cfg = pem_cfg or {}

    def fit(self, model, x, u):
        print("=== Stage 1: MAP ===")
        map_est = MapEstimator(model, **self.map_cfg)
        model = map_est.fit(x, u)
        print("=== Stage 2: PEM ===")
        pem_est = ParticleEMEstimator(model, **self.pem_cfg)
        model = pem_est.fit(x, u)
        return model