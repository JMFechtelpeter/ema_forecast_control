import numpy as np
from itertools import combinations
from scipy.stats import rankdata, norm
import pandas as pd

# ---------- Weighted stats ----------
def weighted_mean(a, w):
    return np.sum(w*a) / np.sum(w)

def weighted_corr(x, y, w):
    # Weighted Pearson
    if len(w)==0:
        return np.nan
    xm = weighted_mean(x, w); ym = weighted_mean(y, w)
    cov = np.sum(w*(x-xm)*(y-ym)) / np.sum(w)
    vx  = np.sum(w*(x-xm)**2)     / np.sum(w)
    vy  = np.sum(w*(y-ym)**2)     / np.sum(w)
    if vx*vy == 0:
        return np.nan
    return cov / np.sqrt(vx*vy)

def weighted_spearman(x, y, w):
    rx = rankdata(x, method='average')
    ry = rankdata(y, method='average')
    return weighted_corr(rx, ry, w)

def weighted_kendall(x, y, w):
    C = D = T = 0.0
    n = len(x)
    for i, j in combinations(range(n), 2):
        wij = w[i]*w[j]
        s = np.sign(x[i]-x[j]) * np.sign(y[i]-y[j])
        if s > 0: C += wij
        elif s < 0: D += wij
        T += wij
    return (C - D) / T


def ci_pvalue_weighted_rank_onepass(xbar, xhat, w, method="spearman",
                                    B=10000, alpha=0.05, alternative='two-sided', seed=0,
                                    p_method="bootstrap", ci_type='percentile', theta0 = 0.0, clip_ci=False,
                                    nan_policy="propagate"):  # "bootstrap" or "permutation"
    rng = np.random.default_rng(seed)
    xbar = np.asarray(xbar); xhat = np.asarray(xhat); w = np.asarray(w)
    if nan_policy == "omit":
        mask = np.isfinite(xbar) & np.isfinite(xhat) & np.isfinite(w)
        xbar = xbar[mask]
        xhat = xhat[mask]
        w = w[mask]
    elif nan_policy == "raise":
        if not (np.isfinite(xbar).all() and np.isfinite(xhat).all() and np.isfinite(w).all()):
            raise ValueError("Input contains NaN or Inf")
    elif nan_policy == "propagate":
        if not (np.isfinite(xbar).all() and np.isfinite(xhat).all() and np.isfinite(w).all()):
            return {
                "statistic": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "p_method": None,
                "B": B,
                "alpha": alpha,
                "alternative": alternative,
                "theta0": theta0,
                "method": f"weighted {method.lower()}"
            }
    n = len(xbar)
    if method.lower() == 'pearson':
        stat_fn = weighted_corr
    elif method.lower().startswith("spear"):
        stat_fn = weighted_spearman
    elif method.lower().startswith("kendall"):
        stat_fn = weighted_kendall
    else:
        raise ValueError(f"Unknown method: {method}")
    if w is None:
        w = np.ones_like(xbar)

    obs = stat_fn(xbar, xhat, w)

    boot_stats = np.empty(B)
    perm_stats = np.empty(B) if p_method=="permutation" else None

    for b in range(B):
        # Bootstrap (groups) for CI and bootstrap-based p
        idx = rng.integers(0, n, n)
        boot_stats[b] = stat_fn(xbar[idx], xhat[idx], w[idx])

        # Optional: permutation stream for permutation p
        if p_method == "permutation":
            perm = rng.permutation(n)
            # weights stay with groups; permute predictions
            perm_stats[b] = stat_fn(xbar, xhat[perm], w)

    # Percentile CI    

    # ---- CI (choose flavor) ----
    q_lo, q_hi = np.percentile(boot_stats, [100*alpha/2, 100*(1-alpha/2)])
    if ci_type == "percentile":
        lo, hi = (q_lo, q_hi)
    elif ci_type == "basic":
        # basic CI: [2*obs - q_hi, 2*obs - q_lo]
        lo, hi = (2*obs - q_hi, 2*obs - q_lo)
        if clip_ci:
            lo = max(lo, -1.0)
            hi = min(hi, 1.0)
    else:
        raise ValueError("ci_type must be 'percentile' or 'basic'.")
    lo = np.min((lo, obs))
    hi = np.max((hi, obs))

    # ---- p-value (matched to CI type, unless permutation) ----
    alt = alternative.lower()
    if p_method == "permutation":
        if alt == "two-sided":
            p = (np.sum(np.abs(perm_stats - theta0) >= np.abs(obs - theta0)) + 1) / (B + 1)
        elif alt == "greater":
            p = (np.sum(perm_stats >= obs) + 1) / (B + 1)
        elif alt == "less":
            p = (np.sum(perm_stats <= obs) + 1) / (B + 1)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        p_note = "permutation p-value (null: no association)"
    else:
        if ci_type == "percentile":
            # Percentile-aligned p: compare theta0 directly to bootstrap dist
            left  = np.mean(boot_stats <= theta0)
            right = np.mean(boot_stats >= theta0)
            if alt == "two-sided":
                p = 2 * min(left, right)
            elif alt == "greater":
                p = right
            elif alt == "less":
                p = left
            else:
                raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
            p_note = "bootstrap percentile p-value (aligned with percentile CI)"
        else:  # basic
            # Basic-aligned p: pivot via cutoff = 2*obs - theta0
            cutoff = 2*obs - theta0
            left  = np.mean(boot_stats <= cutoff)
            right = np.mean(boot_stats >= cutoff)
            if alt == "two-sided":
                p = 2 * min(left, right)
            elif alt == "greater":
                p = right
            elif alt == "less":
                p = left
            else:
                raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
            p_note = "bootstrap basic p-value (aligned with basic CI)"
        p = float(min(1.0, max(0.0, p)))  # clamp to [0,1]


    return {
        "statistic": obs,
        "ci_lower": lo,
        "ci_upper": hi,
        "p_value": float(p),
        "p_method": p_note,
        "B": B,
        "alpha": alpha,
        "alternative": alternative,
        "theta0": theta0,
        "method": f"weighted {method.lower()}"
    }


def weight_corr_for_dataframes(df1, df2, weights, method="spearman", axis=0, **kwargs):
    assert df1.shape == df2.shape
    if weights is None:
        weights = pd.DataFrame(1, index=df1.index, columns=df1.columns)
    assert df1.shape[0] == weights.shape[0]
    if axis == 1:
        df1 = df1.T
        df2 = df2.T
        weights = weights.T
    results = pd.DataFrame(index=df1.columns, columns=['statistic', 'ci_lower', 'ci_upper', 'p_value'])    
    for col in df1.columns:
        res = ci_pvalue_weighted_rank_onepass(df1[col], df2[col], weights[col], method=method, **kwargs)
        results.loc[col] = [res['statistic'], res['ci_lower'], res['ci_upper'], res['p_value']]
    return results


def meta_spearman(rhos, m, alpha=0.05):
    rhos = np.asarray(rhos, float)
    if m is None:
        m = np.full_like(rhos, 8, dtype=float)  # number of groups per participant
    z = np.arctanh(np.clip(rhos, -0.999999, 0.999999))
    w = np.maximum(m - 3, 1e-9)                # inverse Var(z)
    zbar = np.sum(w * z) / np.sum(w)
    se = 1 / np.sqrt(np.sum(w))
    zlo, zhi = zbar + norm.ppf([alpha/2, 1-alpha/2]) * se
    r_pooled = np.tanh(zbar)
    ci = (np.tanh(zlo), np.tanh(zhi))
    zstat = zbar / se
    p = 2 * (1 - norm.cdf(abs(zstat)))
    return dict(effect=r_pooled, ci_lower=ci[0], ci_upper=ci[1], p=p)



def hierarchical_bootstrap_pooled_corr(
    xbar_list,
    xhat_list,
    w_list,
    stat_fn,
    B_outer=10000,
    alpha=0.05,
    seed=None,
    use_fisher_z=True,
):
    """
    Hierarchical bootstrap for a pooled group-level correlation.

    Parameters
    ----------
    xbar_list : list of array-like
        List of length P (participants). Each element is an array of observed
        group means for that participant.
    xhat_list : list of array-like
        Same structure as xbar_list, but for predicted group means.
    w_list : list of array-like
        Same structure as xbar_list, but for group weights (e.g. n per group).
    stat_fn : callable
        Function with signature stat_fn(xbar, xhat, w) that returns a single
        weighted correlation (Spearman, Kendall, etc.) for one participant.
    B_outer : int
        Number of hierarchical bootstrap replicates.
    alpha : float
        1 - confidence level (e.g. 0.05 for 95% CI).
    seed : int
        Random seed for reproducibility.
    use_fisher_z : bool
        If True, pool correlations via mean Fisher-z, then back-transform.
        If False, pool directly via mean correlation.

    Returns
    -------
    result : dict
        {
          "pooled_effect": pooled correlation (on correlation scale),
          "ci_lower": lower CI bound,
          "ci_upper": upper CI bound,
          "alpha": alpha,
          "B_outer": B_outer,
          "use_fisher_z": use_fisher_z,
          "pooled_scale": "correlation",
          "p_value": bootstrap two-sided p for H0: pooled_effect = 0,
          "boot_distribution": np.ndarray of pooled stats on pooling scale
                               (z if use_fisher_z else r),
          "r_per_participant": np.ndarray of observed per-participant correlations
        }
    """
    rng = np.random.default_rng(seed)
    P = len(xbar_list)
    if not (len(xhat_list) == P and len(w_list) == P):
        raise ValueError("xbar_list, xhat_list, w_list must have same length (num participants).")

    # 1) Observed per-participant correlations
    r_obs = []
    for xb, xh, w in zip(xbar_list, xhat_list, w_list):
        xb = np.asarray(xb); xh = np.asarray(xh); w = np.asarray(w)
        mask = np.isfinite(xb) & np.isfinite(xh) & np.isfinite(w)
        xb = xb[mask]
        xh = xh[mask]
        w = w[mask]
        if not (len(xb) == len(xh) == len(w)):
            raise ValueError("Within a participant, xbar, xhat, w must have same length.")
        r_subject = stat_fn(xb, xh, w)
        if np.isfinite(r_subject):
            r_obs.append(r_subject)
    r_obs = np.array(r_obs, float)

    # 2) Observed pooled statistic (mean Fisher-z or mean r)
    if use_fisher_z:
        r_clipped = np.clip(r_obs, -0.999999, 0.999999)
        z_obs = np.arctanh(r_clipped)
        pooled_obs = z_obs.mean()
    else:
        pooled_obs = r_obs.mean()

    # 3) Hierarchical bootstrap: resample participants, then resample groups within
    pooled_boot = np.empty(B_outer, float)

    for b in range(B_outer):
        # resample participants with replacement
        idx_p = rng.integers(0, P, P)
        r_rep = []
        for p in idx_p:
            xb = np.asarray(xbar_list[p])
            xh = np.asarray(xhat_list[p])
            w = np.asarray(w_list[p])
            n_g = len(xb)
            # resample groups with replacement within this participant
            idx_g = rng.integers(0, n_g, n_g)
            r_sample = stat_fn(xb[idx_g], xh[idx_g], w[idx_g])
            if np.isfinite(r_sample):
                r_rep.append(r_sample)
        r_rep = np.array(r_rep, float)

        if use_fisher_z:
            r_rep = np.clip(r_rep, -0.999999, 0.999999)
            z_rep = np.arctanh(r_rep)
            pooled_boot[b] = z_rep.mean()
        else:
            pooled_boot[b] = r_rep.mean()

    # 4) CI on pooled statistic (on pooling scale)
    q_lo, q_hi = np.percentile(pooled_boot, [100*alpha/2, 100*(1-alpha/2)])
    if use_fisher_z:
        pooled_effect = np.tanh(pooled_obs)
        ci_lower = np.tanh(q_lo)
        ci_upper = np.tanh(q_hi)
    else:
        pooled_effect = pooled_obs
        ci_lower, ci_upper = q_lo, q_hi

    # 5) Bootstrap p-value for H0: pooled_effect = 0
    theta0 = 0.0  # 0 correlation => 0 Fisher-z as well
    left  = np.mean(pooled_boot <= theta0)
    right = np.mean(pooled_boot >= theta0)
    p_two_sided = 2 * min(left, right)
    p_two_sided = float(min(1.0, max(0.0, p_two_sided)))

    return {
        "pooled_effect": pooled_effect,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha,
        "B_outer": B_outer,
        "use_fisher_z": use_fisher_z,
        "pooled_scale": "correlation",
        "p_value": p_two_sided,
        "boot_distribution": pooled_boot,
        "r_per_participant": r_obs,
    }


def participant_bootstrap_pooled_corr(
    rhos,
    B=10000,
    alpha=0.05,
    seed=0,
    use_fisher_z=True,
):
    """
    Simple bootstrap over participants only (no within-participant resampling).

    Input is a vector of per-participant correlations (already weighted within
    each participant). The bootstrap resamples participants and computes a
    pooled effect each time.

    Parameters
    ----------
    rhos : array-like
        Per-participant correlations (e.g. weighted Spearman values).
    B : int
        Number of bootstrap replicates.
    alpha : float
        1 - confidence level.
    seed : int
        Random seed.
    use_fisher_z : bool
        If True, pool via mean Fisher-z, else mean correlation.

    Returns
    -------
    result : dict
        {
          "pooled_effect": pooled correlation,
          "ci_lower": lower CI bound,
          "ci_upper": upper CI bound,
          "alpha": alpha,
          "B": B,
          "use_fisher_z": use_fisher_z,
          "pooled_scale": "correlation",
          "p_value": bootstrap two-sided p for H0: pooled_effect = 0,
          "boot_distribution": np.ndarray of pooled stats on pooling scale
        }
    """
    rng = np.random.default_rng(seed)
    rhos = np.asarray(rhos, float)
    P = len(rhos)
    if P == 0:
        raise ValueError("rhos must contain at least one value.")

    # Observed pooled stat
    if use_fisher_z:
        r_clipped = np.clip(rhos, -0.999999, 0.999999)
        z = np.arctanh(r_clipped)
        pooled_obs = z.mean()
    else:
        pooled_obs = rhos.mean()

    # Bootstrap over participants
    pooled_boot = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, P, P)
        r_rep = rhos[idx]
        if use_fisher_z:
            r_rep = np.clip(r_rep, -0.999999, 0.999999)
            z_rep = np.arctanh(r_rep)
            pooled_boot[b] = z_rep.mean()
        else:
            pooled_boot[b] = r_rep.mean()

    # CI
    q_lo, q_hi = np.percentile(pooled_boot, [100*alpha/2, 100*(1-alpha/2)])
    if use_fisher_z:
        pooled_effect = np.tanh(pooled_obs)
        ci_lower = np.tanh(q_lo)
        ci_upper = np.tanh(q_hi)
    else:
        pooled_effect = pooled_obs
        ci_lower, ci_upper = q_lo, q_hi

    # Bootstrap p-value for H0: pooled_effect = 0
    theta0 = 0.0
    left  = np.mean(pooled_boot <= theta0)
    right = np.mean(pooled_boot >= theta0)
    p_two_sided = 2 * min(left, right)
    p_two_sided = float(min(1.0, max(0.0, p_two_sided)))

    return {
        "pooled_effect": pooled_effect,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha,
        "B": B,
        "use_fisher_z": use_fisher_z,
        "pooled_scale": "correlation",
        "p_value": p_two_sided,
        "boot_distribution": pooled_boot,
    }
