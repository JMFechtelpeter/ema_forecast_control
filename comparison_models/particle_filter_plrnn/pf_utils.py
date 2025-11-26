import torch as tc
import math

def masked_log_diag_gauss(x: tc.Tensor, mean: tc.Tensor, std: tc.Tensor) -> tc.Tensor:
    """
    Log N(x; mean, diag(std^2)) using only observed (non-NaN) entries per row.
    x, mean: (N, p) (or (p,) broadcast); std: (p,)
    Returns (N,) log-densities. If a row has p_obs=0, returns 0.0 for that row.
    """
    # mask: True where observed
    obs = ~tc.isnan(x)
    if x.dim() == 1:  # (p,)
        x = x.unsqueeze(0).expand(mean.shape[0], -1)
        obs = obs.unsqueeze(0).expand_as(x)

    # select observed entries
    x_obs    = tc.where(obs, x, tc.zeros_like(x))
    mean_obs = tc.where(obs, mean, tc.zeros_like(mean))
    std_obs  = std  # broadcast later

    # squared residuals only where observed
    resid2 = ((x_obs - mean_obs) / std_obs)**2
    resid2 = tc.where(obs, resid2, tc.zeros_like(resid2))

    # per-row counts and log|R|
    p_obs = obs.sum(dim=1)                                  # (N,)
    log_det = (2.0 * tc.log(std).unsqueeze(0) * obs).sum(dim=1)  # (N,)
    quad = resid2.sum(dim=1)

    # rows with no observations → loglik contribution 0.0
    loglik = tc.where(
        p_obs > 0,
        -0.5 * (quad + log_det + p_obs.to(x.dtype) * math.log(2.0 * math.pi)),
        tc.zeros_like(quad)
    )
    return loglik  # (N,)

def masked_log_diag_gauss_single(x_vec: tc.Tensor, mean_vec: tc.Tensor, std: tc.Tensor) -> tc.Tensor:
    """
    x_vec, mean_vec: (p,) , std: (p,)
    Returns scalar loglik using observed dims; 0 if all missing.
    """
    obs = ~tc.isnan(x_vec)
    p_obs = int(obs.sum().item())
    if p_obs == 0:
        return tc.tensor(0.0, dtype=x_vec.dtype, device=x_vec.device)
    resid = (x_vec[obs] - mean_vec[obs]) / std[obs]
    log_det = (2.0 * tc.log(std[obs])).sum()
    return -0.5 * (resid.pow(2).sum() + log_det + p_obs * math.log(2.0 * math.pi))

def systematic_resample(weights: tc.Tensor, gen: tc.Generator | None = None):
    """
    Systematic resampling (1D). Robust to floating-point edges.
    weights: shape (N,), nonnegative (need not be normalized)
    returns: indices in [0, N-1], shape (N,)
    """
    assert weights.dim() == 1, "weights must be 1D"
    N = weights.numel()
    device, dtype = weights.device, weights.dtype

    # clean & normalize
    w = weights.clone()
    w[~tc.isfinite(w)] = 0
    total = w.sum()
    if total <= 0:
        # fallback: uniform indices
        return tc.arange(N, device=device, dtype=tc.long)
    w = w / total

    # cumulative, force last element to 1.0 exactly
    cdf = tc.cumsum(w, dim=0)
    cdf[-1] = 1.0

    # stratified grid with random offset u0 ∈ [0, 1/N)
    u0 = tc.rand((), generator=gen, device=device, dtype=dtype) / N
    u = u0 + (tc.arange(N, device=device, dtype=dtype) / N)

    # search with right=True to pick the upper bin edge when u==cdf[k]
    idx = tc.searchsorted(cdf, u, right=True)

    # clamp just in case numerical noise pushes to N
    idx = idx.clamp_(max=N - 1).to(tc.long)
    return idx