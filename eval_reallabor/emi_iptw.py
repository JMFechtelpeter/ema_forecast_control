
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

def _time_to_cyc(time_str: pd.Series) -> pd.DataFrame:
    sec = pd.to_timedelta(time_str).dt.total_seconds()
    day = 24 * 3600.0
    return pd.DataFrame({
        "time_sin": np.sin(2 * np.pi * sec / day),
        "time_cos": np.cos(2 * np.pi * sec / day),
    }, index=time_str.index)

def build_intervals_for_iptw(
    df: pd.DataFrame,
    id_col: Optional[str] = None,
    interactive_cols: Optional[List[str]] = None,
    drop_multi_emis: bool = True,
) -> pd.DataFrame:
    if interactive_cols is None:
        interactive_cols = [c for c in df.columns if c.startswith("interactive")]
    needed = ["score", "24h_score", "time", "daynr"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    for c in interactive_cols:
        if c not in df.columns:
            raise ValueError(f"Missing interactive column: {c}")

    if id_col is not None:
        if id_col not in df.columns:
            raise ValueError(f"id_col '{id_col}' not in dataframe.")
        df = df.sort_values([id_col, "daynr", "time"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    cyc = _time_to_cyc(df["time"])
    cov = pd.concat([
        df[["score", "24h_score", "daynr"]].reset_index(drop=True),
        cyc.reset_index(drop=True)
    ], axis=1)

    inter_mat = df[interactive_cols].fillna(0).astype(int).reset_index(drop=True)
    has_any = (inter_mat.sum(axis=1) > 0).astype(int)
    one_hot_idx = (inter_mat > 0).sum(axis=1)

    delta = (df["score"].shift(-1) - df["score"]).reset_index(drop=True)
    valid = ~delta.isna()
    cov = cov.loc[valid].reset_index(drop=True)
    inter_mat = inter_mat.loc[valid].reset_index(drop=True)
    has_any = has_any.loc[valid].reset_index(drop=True)
    one_hot_idx = one_hot_idx.loc[valid].reset_index(drop=True)
    delta = delta.loc[valid].reset_index(drop=True)
    ids = df[id_col].loc[valid].reset_index(drop=True) if id_col is not None else pd.Series(np.zeros(len(delta), dtype=int))

    if drop_multi_emis:
        keep = (one_hot_idx <= 1)
        cov, inter_mat, has_any, one_hot_idx, delta, ids = [obj.loc[keep].reset_index(drop=True) for obj in (cov, inter_mat, has_any, one_hot_idx, delta, ids)]

    t_codes = np.zeros(len(delta), dtype=int)
    if inter_mat.shape[1] > 0:
        for j, col in enumerate(interactive_cols, start=1):
            t_codes[ inter_mat[col].values > 0 ] = j
    T = t_codes  # 0 if none

    out = pd.DataFrame({
        "delta": delta.astype(float),
        "cov_score": cov["score"].astype(float),
        "cov_24h": cov["24h_score"].astype(float),
        "cov_daynr": cov["daynr"].astype(float),
        "cov_time_sin": cov["time_sin"].astype(float),
        "cov_time_cos": cov["time_cos"].astype(float),
        "T": T.astype(int),
        "A": (T>0).astype(int),
        "id": ids if id_col is not None else 0,
    })
    out.attrs["interactive_cols"] = interactive_cols
    return out

@dataclass
class WeightSummary:
    mean: float
    sd: float
    min: float
    p1: float
    p5: float
    p50: float
    p95: float
    p99: float
    max: float
    ess: float

def summarize_weights(w: np.ndarray) -> WeightSummary:
    w = np.asarray(w, dtype=float)
    ess = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-12)
    q = np.percentile(w, [1,5,50,95,99])
    return WeightSummary(
        mean=float(np.mean(w)),
        sd=float(np.std(w)),
        min=float(np.min(w)),
        p1=float(q[0]), p5=float(q[1]), p50=float(q[2]), p95=float(q[3]), p99=float(q[4]),
        max=float(np.max(w)),
        ess=float(ess),
    )

def truncate_weights(w: np.ndarray, lower: float=None, upper: float=None) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    if lower is not None:
        w[w < lower] = lower
    if upper is not None:
        w[w > upper] = upper
    return w

@dataclass
class MultiArmIPTWResult:
    coef: pd.DataFrame
    weight_summary: WeightSummary
    n: int
    n_treated: int
    n_types: int
    dropped_multi_emis: int

def iptw_multiarm_msm(
    inter: pd.DataFrame,
    id_col: str = "id",
    weight_trunc: Tuple[Optional[float], Optional[float]] = (None, None),
    use_wls: bool = True,
    wls_cov_type: str = "HAC",
) -> MultiArmIPTWResult:
    """
    Multi-arm IPTW MSM with robust handling of missing treatment categories.
    Ensures probabilities align to 0..K_total classes (0 = none), assigning zero
    probability to absent classes.
    """
    y = inter["delta"].values.astype(float)
    X_cov = inter[["cov_score","cov_24h","cov_daynr","cov_time_sin","cov_time_cos"]].values.astype(float)
    T = inter["T"].values.astype(int)

    # Total EMI types from attrs if available
    if hasattr(inter, 'attrs') and 'interactive_cols' in inter.attrs:
        K_total = len(inter.attrs['interactive_cols'])
    else:
        K_total = int(np.max(T))

    if K_total < 1:
        raise ValueError("No EMI type columns detected.")
    if not np.any(T == 0):
        raise ValueError("No control (T=0) intervals; need both treated and control.")

    # Multinomial PS on present classes only
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_cov)
    mlogit = LogisticRegression(max_iter=2000)
    mlogit.fit(Xs, T)
    ps_present = mlogit.predict_proba(Xs)   # (n, len(mlogit.classes_))
    n = len(T)
    ps_full = np.zeros((n, K_total + 1), dtype=float)
    for j, c in enumerate(mlogit.classes_):
        ps_full[:, int(c)] = ps_present[:, j]
    # Marginal probs over full label set
    marg = np.bincount(T, minlength=K_total+1) / n

    # Stabilized weights
    eps = 1e-6
    denom = ps_full[np.arange(n), T].clip(eps, 1.0)
    numer = marg[T].clip(eps, 1.0)
    w = numer / denom

    # Truncate if requested
    low, high = weight_trunc
    if low is not None or high is not None:
        w = truncate_weights(w, lower=low, upper=high)

    # Outcome model: delta ~ 1 + I[T=1]..I[T=K_total] (ref T=0)
    dummies = pd.get_dummies(T, prefix="T", drop_first=True, dtype=int)
    for j in range(1, K_total+1):
        col = f"T_{j}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"T_{j}" for j in range(1, K_total+1)]]
    X = sm.add_constant(dummies)

    if use_wls:
        model = sm.WLS(y, X.values, weights=w)
        if id_col in inter.columns:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": inter[id_col].values})
        elif wls_cov_type == "HAC":
            res = model.fit(cov_type=wls_cov_type, cov_kwds={"maxlags":1})
        else:
            res = model.fit(cov_type=wls_cov_type)
    else:
        groups = inter[id_col].values if id_col in inter.columns else np.arange(len(y))
        model = sm.GEE(y, X.values, groups=groups, family=sm.families.Gaussian(), freq_weights=w)
        res = model.fit()

    coefs = res.params
    ses = res.bse
    pvals = res.pvalues if hasattr(res, "pvalues") else np.full_like(coefs, np.nan, dtype=float)

    rows = []
    for j in range(1, K_total+1):
        idx = list(X.columns).index(f"T_{j}")
        beta = float(coefs[idx])
        se = float(ses[idx])
        p = float(pvals[idx]) if np.ndim(pvals)>0 else np.nan
        ci = (beta - 1.96*se, beta + 1.96*se)
        rows.append({"type": j, "effect_vs_none": beta, "se": se, "ci_low": ci[0], "ci_high": ci[1], "p": p})

    ws = summarize_weights(w)
    n_treated = int((T>0).sum())
    return MultiArmIPTWResult(
        coef=pd.DataFrame(rows),
        weight_summary=ws,
        n=n,
        n_treated=n_treated,
        n_types=K_total,
        dropped_multi_emis=int((inter['A']==1).sum() - n_treated) if 'A' in inter.columns else 0
    )

@dataclass
class AnyEMIIPTWResult:
    coef: pd.DataFrame
    weight_summary: WeightSummary
    n: int
    n_treated: int

def iptw_any_emi_with_type_mods(
    inter: pd.DataFrame,
    id_col: str = "id",
    ref_type: int = 1,
    weight_trunc: Tuple[Optional[float], Optional[float]] = (None, None),
    include_covariates_in_outcome: bool = False,
    use_wls: bool = True,
) -> AnyEMIIPTWResult:
    y = inter["delta"].values.astype(float)
    X_cov = inter[["cov_score","cov_24h","cov_daynr","cov_time_sin","cov_time_cos"]].values.astype(float)
    A = inter["A"].values.astype(int)
    T = inter["T"].values.astype(int)
    K = int(T.max())

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_cov)
    logit = LogisticRegression(max_iter=2000)
    logit.fit(Xs, A)
    ps = logit.predict_proba(Xs)[:,1]
    eps = 1e-6
    pA = A.mean().clip(eps,1-eps)
    denom = np.where(A==1, ps.clip(eps,1-eps), (1-ps).clip(eps,1-eps))
    numer = np.where(A==1, pA, 1-pA)
    w = numer / denom

    low, high = weight_trunc
    if low is not None or high is not None:
        w = truncate_weights(w, lower=low, upper=high)

    type_dummies = pd.get_dummies(T, prefix="type")
    for c in type_dummies.columns:
        type_dummies.loc[A==0, c] = 0
    ref_col = f"type_{ref_type}"
    if ref_col in type_dummies.columns:
        type_dummies = type_dummies.drop(columns=[ref_col])

    pieces = [pd.Series(1.0, index=inter.index, name="const"), pd.Series(A, name="A")]
    if include_covariates_in_outcome:
        cov_df = pd.DataFrame(X_cov, columns=["cov_score","cov_24h","cov_daynr","cov_time_sin","cov_time_cos"], index=inter.index)
        pieces.append(cov_df)
    if K > 0 and len(type_dummies.columns) > 0:
        pieces.append(type_dummies)
    X = pd.concat(pieces, axis=1)

    if use_wls:
        model = sm.WLS(y, X.values, weights=w)
        if id_col in inter.columns:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": inter[id_col].values})
        else:
            res = model.fit(cov_type="HC1")
    else:
        groups = inter[id_col].values if id_col in inter.columns else np.arange(len(y))
        model = sm.GEE(y, X.values, groups=groups, family=sm.families.Gaussian(), freq_weights=w)
        res = model.fit()

    names = list(X.columns)
    coefs = res.params
    ses = res.bse
    pvals = res.pvalues if hasattr(res, "pvalues") else np.full_like(coefs, np.nan, dtype=float)

    rows = []
    for name, beta, se, p in zip(names, coefs, ses, pvals):
        if name == "const":
            continue
        ci = (beta - 1.96*se, beta + 1.96*se)
        rows.append({"term": name, "coef": float(beta), "se": float(se), "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p": float(p)})

    ws = summarize_weights(w)
    return AnyEMIIPTWResult(
        coef=pd.DataFrame(rows),
        weight_summary=ws,
        n=len(A),
        n_treated=int(A.sum()),
    )
