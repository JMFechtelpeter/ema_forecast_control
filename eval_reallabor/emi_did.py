
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Literal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

Method = Literal["propensity", "mahalanobis"]

@dataclass
class DIDResult:
    emi_col: str
    n_treated: int
    n_controls_pool: int
    n_matched_treated: int
    n_matched_pairs: int
    att: float
    se: float
    ci95_low: float
    ci95_high: float
    caliper: Optional[float]
    method: Method
    k: int
    dropped_for_no_match: int
    smd_after_matching: Dict[str, float]

def _time_to_cyc(time_str: pd.Series) -> pd.DataFrame:
    sec = pd.to_timedelta(time_str).dt.total_seconds()
    day = 24 * 3600.0
    return pd.DataFrame({
        "time_sin": np.sin(2 * np.pi * sec / day),
        "time_cos": np.cos(2 * np.pi * sec / day),
    }, index=time_str.index)

def _build_intervals(df: pd.DataFrame, id_col: Optional[str], emi_col: str, interactive_cols: List[str], exclude_other_emis: bool=True) -> pd.DataFrame:
    needed = ["score", "24h_score", "time", "daynr"]
    for c in needed + [emi_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' is missing.")
    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not in dataframe.")
    time_feats = _time_to_cyc(df["time"])
    # Build intervals within each id (or globally if id_col is None)
    if id_col is None:
        groups = [df.reset_index(drop=True)]
        group_keys = [None]
    else:
        groups = [g.reset_index(drop=True) for _, g in df.sort_values([id_col, "daynr", "time"]).groupby(id_col)]
        group_keys = [g.iloc[0][id_col] for g in groups]
    rows = []
    for g_key, g in zip(group_keys, groups):
        n = len(g)
        if n < 2:
            continue
        cov_g = pd.concat([g[["score", "24h_score", "daynr"]].reset_index(drop=True),
                           _time_to_cyc(g["time"]).reset_index(drop=True)], axis=1)
        T = (g[emi_col].fillna(0) > 0).astype(int).values[:-1]
        mask_other = np.ones(n-1, dtype=bool)
        if exclude_other_emis:
            others = [c for c in interactive_cols if c != emi_col and c in g.columns]
            if others:
                mask_other = (g[others].iloc[:-1].fillna(0).sum(axis=1) == 0).values
        delta = (g["score"].shift(-1) - g["score"]).values[:-1]
        keep = ~pd.isna(delta) & mask_other
        for i in np.where(keep)[0]:
            rows.append({
                "id": g_key if id_col is not None else 0,
                "cov_score": float(cov_g.at[i, "score"]),
                "cov_24h": float(cov_g.at[i, "24h_score"]),
                "cov_daynr": float(cov_g.at[i, "daynr"]),
                "cov_time_sin": float(cov_g.at[i, "time_sin"]),
                "cov_time_cos": float(cov_g.at[i, "time_cos"]),
                "T": int(T[i]),
                "delta": float(delta[i]),
            })
    inter = pd.DataFrame(rows)
    return inter

def _standardized_mean_diff(x_t: np.ndarray, x_c: np.ndarray, w_t: np.ndarray, w_c: np.ndarray) -> float:
    xt = np.average(x_t, weights=w_t)
    xc = np.average(x_c, weights=w_c)
    vt = np.average((x_t-xt)**2, weights=w_t)
    vc = np.average((x_c-xc)**2, weights=w_c)
    sd_pooled = np.sqrt(0.5*(vt+vc) + 1e-12)
    return float((xt - xc)/sd_pooled)

def _compute_smds(inter: pd.DataFrame, matched_idx: List[Tuple[int,int]], k: int) -> Dict[str, float]:
    if not matched_idx:
        return {}
    covs = ["cov_score", "cov_24h", "cov_daynr", "cov_time_sin", "cov_time_cos"]
    treat_rows, treat_w, ctrl_rows, ctrl_w = [], [], [], []
    for (i_t, i_c) in matched_idx:
        treat_rows.append(inter.loc[i_t, covs].values.astype(float))
        treat_w.append(1.0)
        ctrl_rows.append(inter.loc[i_c, covs].values.astype(float))
        ctrl_w.append(1.0/float(k))
    X_t = np.vstack(treat_rows)
    X_c = np.vstack(ctrl_rows)
    w_t = np.array(treat_w)
    w_c = np.array(ctrl_w)
    smds = {}
    for j, name in enumerate(covs):
        smds[name] = _standardized_mean_diff(X_t[:, j], X_c[:, j], w_t, w_c)
    return smds

def _match_propensity(inter: pd.DataFrame, k: int, caliper: Optional[float]) -> Tuple[List[Tuple[int,int]], int]:
    covs = ["cov_score", "cov_24h", "cov_daynr", "cov_time_sin", "cov_time_cos"]
    X = inter[covs].values.astype(float)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    y = inter["T"].values.astype(int)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(Xs, y)
    p = lr.predict_proba(Xs)[:,1]
    inter = inter.copy()
    inter["ps"] = p
    treated_idx = inter.index[inter["T"]==1].tolist()
    control_idx = inter.index[inter["T"]==0].tolist()
    if len(treated_idx)==0 or len(control_idx)==0:
        return [], len(treated_idx)
    ps_t = inter.loc[treated_idx, "ps"].values
    ps_c = inter.loc[control_idx, "ps"].values
    matched_pairs = []
    dropped = 0
    for it, pval in zip(treated_idx, ps_t):
        dists = np.abs(ps_c - pval)
        order = np.argsort(dists)
        chosen = []
        for idx in order:
            if caliper is not None and dists[idx] > caliper:
                break
            chosen.append(control_idx[idx])
            if len(chosen) == k:
                break
        if len(chosen) < k:
            dropped += 1
            continue
        for ic in chosen:
            matched_pairs.append((it, ic))
    return matched_pairs, dropped

def _match_mahalanobis(inter: pd.DataFrame, k: int, caliper: Optional[float]) -> Tuple[List[Tuple[int,int]], int]:
    covs = ["cov_score", "cov_24h", "cov_daynr", "cov_time_sin", "cov_time_cos"]
    X = inter[covs].values.astype(float)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    inter = inter.copy()
    inter[covs] = Xs
    treated_idx = inter.index[inter["T"]==1].tolist()
    control_idx = inter.index[inter["T"]==0].tolist()
    if len(treated_idx)==0 or len(control_idx)==0:
        return [], len(treated_idx)
    X_c = inter.loc[control_idx, covs].values
    X_t = inter.loc[treated_idx, covs].values
    S = np.cov(X_c, rowvar=False)
    reg = 1e-6 * np.eye(S.shape[0])
    VI = np.linalg.pinv(S + reg)
    matched_pairs = []
    dropped = 0
    for i, it in enumerate(treated_idx):
        diff = X_c - X_t[i]
        d2 = np.einsum("ij,jk,ik->i", diff, VI, diff)
        order = np.argsort(d2)
        chosen = []
        for idx in order:
            d = float(np.sqrt(d2[idx]))
            if caliper is not None and d > caliper:
                break
            chosen.append(control_idx[idx])
            if len(chosen) == k:
                break
        if len(chosen) < k:
            dropped += 1
            continue
        for ic in chosen:
            matched_pairs.append((it, ic))
    return matched_pairs, dropped

def _att_and_se(inter: pd.DataFrame, matched_idx: List[Tuple[int,int]], k: int) -> Tuple[float, float]:
    if not matched_idx:
        return np.nan, np.nan
    from collections import defaultdict
    ctrl_by_t = defaultdict(list)
    for it, ic in matched_idx:
        ctrl_by_t[it].append(ic)
    d_i = []
    for it, ics in ctrl_by_t.items():
        delta_t = inter.loc[it, "delta"]
        delta_c = inter.loc[ics, "delta"].values
        d_i.append(delta_t - np.mean(delta_c))
    d_i = np.array(d_i, dtype=float)
    att = float(np.mean(d_i))
    se = float(np.std(d_i, ddof=1) / np.sqrt(len(d_i))) if len(d_i) > 1 else np.nan
    return att, se

def did_match_single_emi(
    df: pd.DataFrame,
    emi_col: str,
    id_col: Optional[str]=None,
    method: Method = "propensity",
    k: int = 3,
    caliper: Optional[float] = None,
    exclude_other_emis: bool = True,
) -> DIDResult:
    interactive_cols = [c for c in df.columns if c.startswith("interactive")]
    if emi_col not in interactive_cols:
        raise ValueError(f"{emi_col} not found among interactive columns: {interactive_cols}")
    inter = _build_intervals(df, id_col=id_col, emi_col=emi_col, interactive_cols=interactive_cols, exclude_other_emis=exclude_other_emis)
    if inter.empty:
        raise ValueError("No valid intervals could be constructed (check data and id_col).")
    n_treated = int((inter["T"]==1).sum())
    n_controls_pool = int((inter["T"]==0).sum())
    if n_treated == 0 or n_controls_pool == 0:
        raise ValueError("Need both treated and control intervals for matching.")
    if method == "propensity":
        matched_idx, dropped = _match_propensity(inter, k=k, caliper=caliper)
    elif method == "mahalanobis":
        matched_idx, dropped = _match_mahalanobis(inter, k=k, caliper=caliper)
    else:
        raise ValueError("method must be 'propensity' or 'mahalanobis'")
    att, se = _att_and_se(inter, matched_idx, k=k)
    if se is not np.nan and se is not None:
        ci_low = att - 1.96*se if not np.isnan(se) else np.nan
        ci_high = att + 1.96*se if not np.isnan(se) else np.nan
    else:
        ci_low, ci_high = np.nan, np.nan
    matched_treated = len(set([it for it, _ in matched_idx]))
    smds = _compute_smds(inter, matched_idx, k=k)
    return DIDResult(
        emi_col=emi_col,
        n_treated=n_treated,
        n_controls_pool=n_controls_pool,
        n_matched_treated=matched_treated,
        n_matched_pairs=len(matched_idx),
        att=att,
        se=se,
        ci95_low=ci_low,
        ci95_high=ci_high,
        caliper=caliper,
        method=method,
        k=k,
        dropped_for_no_match=dropped,
        smd_after_matching=smds,
    )

def did_match_all_emi(
    df: pd.DataFrame,
    emi_cols: Optional[List[str]]=None,
    id_col: Optional[str]=None,
    method: Method = "propensity",
    k: int = 3,
    caliper: Optional[float] = None,
    exclude_other_emis: bool = True,
) -> pd.DataFrame:
    if emi_cols is None:
        emi_cols = [c for c in df.columns if c.startswith("interactive")]
        emi_cols = sorted(emi_cols, key=lambda x: (len(x), x))
    results: List[DIDResult] = []
    rows = []
    for c in emi_cols:
        try:
            res = did_match_single_emi(
                df=df,
                emi_col=c,
                id_col=id_col,
                method=method,
                k=k,
                caliper=caliper,
                exclude_other_emis=exclude_other_emis,
            )
            smds = res.smd_after_matching
            row = {
                "emi_col": res.emi_col,
                "method": res.method,
                "k": res.k,
                "caliper": res.caliper,
                "n_treated": res.n_treated,
                "n_controls_pool": res.n_controls_pool,
                "n_matched_treated": res.n_matched_treated,
                "n_matched_pairs": res.n_matched_pairs,
                "dropped_for_no_match": res.dropped_for_no_match,
                "att": res.att,
                "se": res.se,
                "ci95_low": res.ci95_low,
                "ci95_high": res.ci95_high,
            }
            for k_smd, v in smds.items():
                row[f"smd_{k_smd}"] = v
            rows.append(row)
        except Exception as e:
            rows.append({
                "emi_col": c, "method": method, "k": k, "caliper": caliper,
                "n_treated": 0, "n_controls_pool": 0, "n_matched_treated": 0, "n_matched_pairs": 0,
                "dropped_for_no_match": 0, "att": np.nan, "se": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                "smd_error": str(e)
            })
    summary = pd.DataFrame(rows)
    # Order columns nicely
    base_cols = ["emi_col","method","k","caliper","n_treated","n_controls_pool","n_matched_treated","n_matched_pairs","dropped_for_no_match","att","se","ci95_low","ci95_high"]
    smd_cols = [c for c in summary.columns if c.startswith("smd_")]
    summary = summary[base_cols + smd_cols]
    return summary
