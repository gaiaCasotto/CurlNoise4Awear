"""""EEG feature extraction and smoothing utilities."""

import numpy as np
import pandas as pd
from scipy.signal import welch

# Define EEG frequency bands
bands = {
    "delta": (0.1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 42),
    "alpha1": (8, 10),
    "alpha2": (10, 12),
    "beta1": (12, 18),
    "beta2": (18, 24),
    "beta3": (24, 30),
    "gamma1": (30, 38),
    "gamma2": (38, 42),
}


def compute_psd(signal: np.ndarray, fs: int):
    """
    Compute the power spectral density (PSD) using Welch's method.

    Parameters
    ----------
    signal : np.ndarray
        Time series EEG segment.
    fs : int
        Sampling frequency.

    Returns
    -------
    freqs : np.ndarray
        Array of frequency bins.
    psd : np.ndarray
        Power spectral density for the signal.
    """
    freqs, psd = welch(signal, fs=fs, nperseg=len(signal), window="hann")
    return freqs, psd


def bandpower(freqs, psd, band):
    """
    Compute bandpower for a given frequency band using the trapezoidal rule.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency bins.
    psd : np.ndarray
        Power spectral density.
    band : tuple
        Lower and upper bound of frequency band.

    Returns
    -------
    float
        Power in the given frequency band.
    """
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapz(psd[mask], freqs[mask])


def extract_band_features(
    freqs,
    psd,
    signal=None,
    document_name=None,
    segment=None,
    focus_type=None,
    session_id=None,
    timestamp=None,
):
    """
    Extract power features for each EEG frequency band, with optional metadata.

    Parameters
    ----------
    freqs : np.ndarray
    psd : np.ndarray
    signal : np.ndarray, optional
    document_name : str, optional
    segment : int, optional
    focus_type : str, optional
    session_id : str, optional
    timestamp : str, optional

    Returns
    -------
    dict
        Feature dictionary with band powers and metadata.
    """
    powers = {name: bandpower(freqs, psd, b) for name, b in bands.items()}

    if document_name is not None:
        powers["document_name"] = document_name
    if segment is not None:
        powers["segment"] = segment
    if focus_type is not None:
        powers["focus_type"] = focus_type
    if session_id is not None:
        powers["session_id"] = session_id
    if timestamp is not None:
        powers["timestamp"] = timestamp
    return powers


def apply_ema_filtering(features_df: pd.DataFrame, alpha: float = 0.9) -> pd.DataFrame:
    """
    Apply exponential moving average (EMA) filtering to band features and compute derived ratios.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame of EEG features.
    alpha : float
        Smoothing factor for EMA.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional EMA-smoothed features and derived indexes.
    """
    filtered_df = features_df.copy()

    band_features = list(bands.keys())
    entropy_features = [col for col in features_df.columns if "entropy" in col]


    if "document_name" in features_df.columns:
        for document_name in filtered_df["document_name"].unique():
            document_name_mask = filtered_df["document_name"] == document_name

            for band in band_features:
                ema_values = filtered_df.loc[document_name_mask, band].ewm(alpha=alpha, adjust=False).mean()
                filtered_df.loc[document_name_mask, f"{band}_fil"] = ema_values

            for entropy in entropy_features:
                ema_values = filtered_df.loc[document_name_mask, entropy].ewm(alpha=alpha, adjust=False).mean()
                filtered_df.loc[document_name_mask, f"{entropy}_fil"] = ema_values
    else:
        for band in band_features:
            ema_values = filtered_df.loc[:, band].ewm(alpha=alpha, adjust=False).mean()
            filtered_df.loc[:, f"{band}_fil"] = ema_values

        for entropy in entropy_features:
            ema_values = filtered_df.loc[:, entropy].ewm(alpha=alpha, adjust=False).mean()
            filtered_df.loc[:, f"{entropy}_fil"] = ema_values
            
    filtered_df["theta_beta_ratio_fil"] = (
        filtered_df["theta_fil"] / filtered_df["beta_fil"]
    )
    filtered_df["theta_alpha_ratio_fil"] = (
        filtered_df["theta_fil"] / filtered_df["alpha_fil"]
    )
    filtered_df["beta_alpha_ratio_fil"] = (
        filtered_df["beta_fil"] / filtered_df["alpha_fil"]
    )
    filtered_df["engagement_index_fil"] = (
        filtered_df["alpha_fil"] / filtered_df["beta_fil"]
    )
    filtered_df["focus_index_fil"] = filtered_df["beta_fil"] / (
        filtered_df["theta_fil"] + filtered_df["alpha_fil"]
    )

    eps = 1e-12
    for name in bands.keys():
        filtered_df[f"{name}_db_fil"] = 10 * np.log10(filtered_df[name+'_fil'] + eps)
    for name in [
        "theta_beta_ratio_fil",
        "theta_alpha_ratio_fil",
        "beta_alpha_ratio_fil",
        "engagement_index_fil",
        "focus_index_fil",
    ]:
        filtered_df[f"{name}_db"] = 10 * np.log10(filtered_df[name] + eps)

    filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)
    filtered_df = filtered_df.fillna(0)
    return filtered_df


def normalize_indexes(
    features_df: pd.DataFrame, columns_to_normalize: list
) -> pd.DataFrame:
    """
    Normalize selected features (z-score) per subject/document_name.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame of features.
    columns_to_normalize : list
        Columns to normalize.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame.
    """
    normalized_df = features_df.copy()

    for document_name in normalized_df["document_name"].unique():
        document_name_mask = normalized_df["document_name"] == document_name

        for column in columns_to_normalize:
            values = normalized_df.loc[document_name_mask, column]
            mean_val = values.mean()
            std_val = values.std()

            if std_val == 0:
                continue

            normalized_df.loc[document_name_mask, f"{column}_norm"] = (
                values - mean_val
            ) / std_val

    return normalized_df


def add_time_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived time-of-day features from timestamp.

    Parameters
    ----------
    features_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    features_df["time_of_day"] = pd.to_datetime(features_df["timestamp"]).dt.time
    features_df["hours_since_midnight"] = (
        pd.to_datetime(features_df["timestamp"]).dt.hour
        + pd.to_datetime(features_df["timestamp"]).dt.minute / 60
        + pd.to_datetime(features_df["timestamp"]).dt.second / 3600
    )
    features_df["minutes_since_midnight"] = (
        pd.to_datetime(features_df["timestamp"]).dt.hour * 60
        + pd.to_datetime(features_df["timestamp"]).dt.minute
    )
    return features_df
