from typing import Any, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu


def cohens_d(x: List[float], y: List[float]) -> float:
    """
    Compute Cohen's d for two independent samples.

    Returns nan if the pooled standard deviation is zero.
    """
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = x_arr.size, y_arr.size
    dof = nx + ny - 2
    var_x = np.var(x_arr, ddof=1)
    var_y = np.var(y_arr, ddof=1)
    pooled_var = ((nx - 1) * var_x + (ny - 1) * var_y) / dof
    pooled_std = np.sqrt(pooled_var)
    return (np.mean(x_arr) - np.mean(y_arr)) / pooled_std if pooled_std > 0 else np.nan


def compare_session_types(
    features_df: pd.DataFrame,
    feature_columns: List[str],
    session_type_col: str = "focus_type",
    document_name_col: str = "document_name",
) -> pd.DataFrame:
    """
    For each document_name, compare two session types on each feature:
      - Kolmogorov–Smirnov
      - Mann–Whitney U
      - Cohen’s d

    Parameters
    ----------
    features_df : pd.DataFrame
        Input data with features and metadata.
    feature_columns : Sequence[str]
        Names of numeric columns to compare.
    session_type_col : str
        Column name for the session type (exactly two unique values).
    document_name_col : str
        Column name to group by (formerly `email`).

    Returns
    -------
    pd.DataFrame
        One row per (document_name, feature) pair with test statistics and sample sizes.
    """
    # Identify the two session types
    session_types = features_df[session_type_col].dropna().unique()
    if session_types.size != 2:
        raise ValueError("Expected exactly two distinct session types.")
    type_a, type_b = session_types

    records = []
    # Group by document_name
    for doc_name, group in features_df.groupby(document_name_col):
        grp_a = group[group[session_type_col] == type_a]
        grp_b = group[group[session_type_col] == type_b]

        for feat in feature_columns:
            a_vals = grp_a[feat].dropna()
            b_vals = grp_b[feat].dropna()
            if a_vals.empty or b_vals.empty:
                continue

            ks_stat, ks_p = ks_2samp(a_vals, b_vals)
            mw_stat, mw_p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            d = abs(cohens_d(a_vals, b_vals))

            records.append(
                {
                    document_name_col: doc_name,
                    "feature": feat,
                    f"{session_type_col}_1": type_a,
                    f"{session_type_col}_2": type_b,
                    "ks_stat": ks_stat,
                    "ks_pvalue": ks_p,
                    "mw_stat": mw_stat,
                    "mw_pvalue": mw_p,
                    "cohens_d": d,
                    "n1": a_vals.size,
                    "n2": b_vals.size,
                }
            )

    return pd.DataFrame(records)
