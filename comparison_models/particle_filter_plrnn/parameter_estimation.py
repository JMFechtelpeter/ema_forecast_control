import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from tqdm import trange
from dataclasses import dataclass
from torch.distributions.multivariate_normal import MultivariateNormal
import math

# ===============================================================
# Base Estimator class
# ===============================================================

class BaseEstimator:
    """Common interface for all estimators."""
    def __init__(self, model):
        self.model = model
        self.theta = {k: v.clone().detach().requires_grad_(True)
                      for k, v in model.params.items()}

    def update_model(self):
        """Write local theta back into model.params"""
        for k, v in self.theta.items():
            self.model.params[k].data = v.data.clone()

    def fit(self, x, u):
        raise NotImplementedError


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

def systematic_resample(weights: Tensor, gen: tc.Generator | None = None):
    N = weights.numel()
    u0 = tc.rand((), generator=gen, device=weights.device, dtype=weights.dtype) / N
    cdf = tc.cumsum(weights, dim=0)
    u = u0 + tc.arange(N, device=weights.device, dtype=weights.dtype) / N
    idx = tc.searchsorted(cdf, u)
    return idx.to(tc.long)

@dataclass
class EStepStats:
    S_zz: Tensor; S_xz: Tensor; S_xx: Tensor
    S_phiphi: list; S_zplus_phi: list
    S_zplus_sq: Tensor; S_mu_sq: Tensor; S_zplus_mu: Tensor
    Ez_T: Tensor; loglik: float; varloglik: float; mse: float

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

    def fit(self, x: Tensor, u: Tensor):
        # standardize x, keep stats on model for forecasting
        x_mean = x.nanmean(dim=0)
        x_std = ((x - x.nanmean(dim=0, keepdim=True))**2).nanmean(dim=0).clamp_min(1e-8)
        x_s = (x - x_mean) / x_std
        self.model.x_mean = x_mean.detach().clone()
        self.model.x_std  = x_std.detach().clone()

        T = x.shape[0]
        T_val = int(T * self.val_split)
        if T_val > 0:
            x_tr, u_tr = x_s[:-T_val], u[:-T_val]
            x_val, u_val = x_s[-T_val:], u[-T_val:]
        else:
            x_tr, u_tr = x_s, u
            x_val = u_val = None

        pbar = trange(self.n_iter, desc="PEM (APF+FFBSi)", leave=True)
        best_snapshot, best_val = None, float("inf")

        for it in pbar:
            stats = self.e_step(x_tr, u_tr)   # ---- E-step (APF + FFBSi)
            self.m_step(stats)                # ---- M-step

            if x_val is not None:
                ll_val, _ = self._apf_loglik(x_val, u_val, self.N)
                nll_val = -ll_val / len(x_val)
                if nll_val < best_val - 1e-4:
                    best_val = nll_val
                    best_snapshot = {k: v.detach().clone() for k,v in self.model.params.items()}
            else:
                nll_val = float('nan')

            pbar.set_description_str(
                f"PEM it={it+1}/{self.n_iter} | loglik={stats.loglik:.2f} "
                f"| varloglik={stats.varloglik:.3f} | MSE={stats.mse:.4f} "
                + (f"| valNLL={nll_val:.4f}" if x_val is not None else "")
            )

        if best_snapshot is not None:
            for k,v in best_snapshot.items():
                self.model.params[k].data = v

        self.last_Ez_T = stats.Ez_T.detach().clone()
        self.update_model()
        return self.model

    # ---------------------------
    # E-step: APF + FFBSi
    # ---------------------------
    def e_step(self, x: Tensor, u: Tensor) -> EStepStats:
        T, p = x.shape
        d, m = self.model.d, self.model.m
        device, dtype = x.device, x.dtype

        # forward APF (store particle clouds/weights for FFBSi)
        zs, ws, znexts, loglik = self._apf_forward_store(x, u, self.N)

        # variance of loglik (diagnostic)
        varloglik = 0.0
        if self.var_runs > 1:
            ll_runs = [self._apf_loglik(x, u, self.N)[0] for _ in range(self.var_runs)]
            varloglik = float(np.var(ll_runs, ddof=1) if len(ll_runs) > 1 else 0.0)

        # accumulators
        S_zz = tc.zeros(d, d, device=device, dtype=dtype)
        S_xz = tc.zeros(p, d, device=device, dtype=dtype)
        S_xx = (x[:, :, None] * x[:, None, :]).sum(dim=0)

        S_phiphi = [tc.zeros(1 + (d-1) + m + 1, 1 + (d-1) + m + 1, device=device, dtype=dtype) for _ in range(d)]
        S_zplus_phi = [tc.zeros(1, 1 + (d-1) + m + 1, device=device, dtype=dtype) for _ in range(d)]

        S_zplus_sq = tc.zeros(d, device=device, dtype=dtype)
        S_mu_sq    = tc.zeros(d, device=device, dtype=dtype)
        S_zplus_mu = tc.zeros(d, device=device, dtype=dtype)

        Ez_T_sum = tc.zeros(d, device=device, dtype=dtype)

        for _ in range(self.M):
            z_path = self._ffbsi_sample_path(zs, ws, znexts, u)   # (T+1,d)

            relu_z = F.relu(z_path[:-1])  # (T,d)
            # sums for emissions (align z_t with x_t at t=1..T using z_path[1:])
            S_zz += (z_path[1:, :, None] * z_path[1:, None, :]).sum(dim=0)
            S_xz += (x[:, :, None] * z_path[1:, None, :]).sum(dim=0)

            for i in range(d):
                relu_mask = [j for j in range(d) if j != i]
                phi_i = tc.cat([
                    z_path[:-1][:, i:i+1],
                    relu_z[:, relu_mask],
                    u,
                    tc.ones(T,1, device=device, dtype=dtype)
                ], dim=1)  # (T, k_i)

                S_phiphi[i]    += phi_i.T @ phi_i
                S_zplus_phi[i] += z_path[1:, i:i+1].T @ phi_i

            mu_next = self.model.f_mean(z_path[:-1], u)   # (T,d)
            S_zplus_sq += (z_path[1:]**2).sum(dim=0)
            S_mu_sq    += (mu_next**2).sum(dim=0)
            S_zplus_mu += (z_path[1:] * mu_next).sum(dim=0)

            Ez_T_sum += z_path[-1]

        Ez_T = Ez_T_sum / float(self.M)

        # quick MSE diagnostic using 3 smoothed paths avg
        Ez_t = (tc.stack([self._ffbsi_sample_path(zs, ws, znexts, u) for _ in range(3)], dim=0).mean(dim=0))
        x_hat = self.model.emission_mean(Ez_t[1:])
        mse = float(((x - x_hat)**2).mean().item())

        return EStepStats(
            S_zz=S_zz, S_xz=S_xz, S_xx=S_xx,
            S_phiphi=S_phiphi, S_zplus_phi=S_zplus_phi,
            S_zplus_sq=S_zplus_sq, S_mu_sq=S_mu_sq, S_zplus_mu=S_zplus_mu,
            Ez_T=Ez_T, loglik=float(loglik), varloglik=float(varloglik), mse=mse
        )

    # ---------------------------
    # M-step (unchanged in spirit; ridge + constraints)
    # ---------------------------
    def m_step(self, stats: EStepStats):
        d, m, p = self.model.d, self.model.m, self.model.p
        device = self.model.params['AW'].device
        dtype  = self.model.params['AW'].dtype

        # B (ridge) and R (diag)
        Szz_reg = stats.S_zz + self.lam_B * tc.eye(d, device=device, dtype=dtype)
        B = stats.S_xz @ tc.linalg.solve(Szz_reg, tc.eye(d, device=device, dtype=dtype))
        Sxx = stats.S_xx
        term = Sxx - 2 * (stats.S_xz @ B.T) + (B @ stats.S_zz @ B.T)
        T_eff = term.shape[0]  # equals T
        R_diag = tc.diag(term) / T_eff
        R_std  = R_diag.clamp_min(self.noise_floor).sqrt()
        self.theta['B'].data = B
        self.theta['R'].data = tc.log(tc.exp(R_std) - 1.0).clamp_min(math.log(self.noise_floor))

        # Dynamics rows
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

        # Q diag
        Q_var = tc.empty(d, device=device, dtype=dtype)
        for i in range(d):
            k_i = stats.S_phiphi[i].shape[0]
            I_k = tc.eye(k_i, device=device, dtype=dtype)
            m_i = stats.S_zplus_phi[i] @ tc.linalg.solve(stats.S_phiphi[i] + self.lam_dyn * I_k, I_k)
            quad  = (m_i @ stats.S_phiphi[i] @ m_i.T).squeeze()
            cross = (m_i @ stats.S_zplus_phi[i].T).squeeze()
            Q_var[i] = (stats.S_zplus_sq[i] - 2 * cross + quad) / T_eff

        Q_std = Q_var.clamp_min(self.noise_floor).sqrt()

        AW = W_off + tc.diag(A_diag)
        self.theta['AW'].data = AW
        self.theta['C'].data  = C_mat
        # if/when you add b as a parameter, write it back here
        self.theta['Q'].data  = tc.log(tc.exp(Q_std) - 1.0).clamp_min(math.log(self.noise_floor))

    def _bound_Aii(self, a_raw: Tensor, mu: float = 0.95, span: float = 0.10):
        return mu + span * tc.tanh(a_raw)

    # ---------------------------
    # Auxiliary PF (APF) internals
    # ---------------------------
    def _apf_forward_store(self, x: Tensor, u: Tensor, N: int):
        """
        APF with look-ahead m_i = p(x_{t+1} | mu_{t+1|t}^i).
        Store pre-resampling clouds/weights for FFBSi.
        Returns: zs, ws, znexts, loglik
        """
        T, p = x.shape
        d = self.model.d
        device, dtype = x.device, x.dtype
        gen = self.gen

        # t=0: prior + weight on x_0
        z_t = self.model.sample_z0(N, rng=gen, device=device, dtype=dtype)  # (N,d)
        logw = self.model.log_p_x_given_z(x[0].expand(N, p), z_t)           # (N,)
        logZ = logsumexp(logw)
        w_t = tc.exp(logw - logZ)
        loglik = float((logZ - math.log(N)).item())

        zs = [z_t.clone()]   # particle cloud at t (pre-resampling)
        ws = [w_t.clone()]
        znexts = []

        for t in range(T - 1):
            # Look-ahead predictive likelihood m_i for x_{t+1}
            mu_next = self.model.f_mean(z_t, u[t])                           # (N,d)
            x_pred  = self.model.emission_mean(mu_next)                      # (N,p)
            r_std   = F.softplus(self.model.params['R']) + 1e-6              # (p,)
            # log p(x_{t+1} | mu_next) with diag R
            log_m = -0.5 * (((x[t+1].expand_as(x_pred) - x_pred) / r_std)**2).sum(dim=1) \
                    - 0.5 * (2*tc.log(r_std).sum() + p*math.log(2*math.pi))  # (N,)
            m_i = tc.exp(log_m - log_m.max())  # safe positive

            # Auxiliary weights r_i ∝ w_t(i) * m_i
            r = w_t * m_i
            r_sum = r.sum()
            r = r / r_sum.clamp_min(1e-32)

            # Resample ancestors ~ r
            idx = systematic_resample(r, gen)
            anc  = z_t[idx]                      # (N,d)
            m_ai = m_i[idx].clamp_min(1e-32)     # (N,)

            # Propagate
            z_prop = self.model.sample_z_next(anc, u[t], rng=gen)            # (N,d)
            # Correction weights: w_corr ∝ p(x_{t+1}|z_prop) / m_{a_i}
            log_num = self.model.log_p_x_given_z(x[t+1].expand(N, p), z_prop)  # (N,)
            log_den = tc.log(m_ai)
            log_wcorr = log_num - log_den
            logZ2 = logsumexp(log_wcorr)
            w_next = tc.exp(log_wcorr - logZ2)

            # Marginal loglik increment: log r_sum + log( (1/N) sum w_corr )
            loglik += float(tc.log(r_sum + 1e-32).item() + (logZ2 - math.log(N)))

            # Store for FFBSi (pre-resampling at t+1)
            znexts.append(z_prop.clone())
            zs.append(z_prop.clone())
            ws.append(w_next.clone())

            # Resample if ESS low for next step
            ess_next = 1.0 / (w_next**2).sum()
            if ess_next < self.ess_thr * N:
                idx2 = systematic_resample(w_next, gen)
                z_t = z_prop[idx2]
                w_t = tc.full_like(w_next, 1.0 / N)
            else:
                z_t = z_prop
                w_t = w_next

        return zs, ws, znexts, loglik

    def _apf_loglik(self, x: Tensor, u: Tensor, N: int):
        """APF marginal log-likelihood only (no storage)."""
        T, p = x.shape
        device, dtype = x.device, x.dtype
        gen = self.gen

        z_t = self.model.sample_z0(N, rng=gen, device=device, dtype=dtype)
        logw = self.model.log_p_x_given_z(x[0].expand(N, p), z_t)
        logZ = logsumexp(logw)
        w = tc.exp(logw - logZ)
        loglik = float((logZ - math.log(N)).item())

        for t in range(T - 1):
            mu_next = self.model.f_mean(z_t, u[t])
            x_pred  = self.model.emission_mean(mu_next)
            r_std   = F.softplus(self.model.params['R']) + 1e-6
            log_m = -0.5 * (((x[t+1].expand_as(x_pred) - x_pred) / r_std)**2).sum(dim=1) \
                    - 0.5 * (2*tc.log(r_std).sum() + p*math.log(2*math.pi))
            m_i = tc.exp(log_m - log_m.max())
            r = w * m_i
            r_sum = r.sum()
            r = r / r_sum.clamp_min(1e-32)

            idx = systematic_resample(r, gen)
            anc = z_t[idx]
            m_ai = m_i[idx].clamp_min(1e-32)

            z_prop = self.model.sample_z_next(anc, u[t], rng=gen)
            log_num = self.model.log_p_x_given_z(x[t+1].expand(N, p), z_prop)
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
