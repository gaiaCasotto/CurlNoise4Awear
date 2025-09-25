import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import zscore


def detect_artifacts(
    seg: np.ndarray,
    fs: int,
    method: str = "amplitude",
    amp_thresh: float = 20.0,
    z_thresh: float = 3.0,
    z_outlier_fraction: float = 0.05,
    gamma_power_thresh: float = None,
    gamma_band: tuple = (30, 47),
    min_gamma_power: float = None,
) -> bool:
    """
    Detect whether a segment contains artifacts using the specified method.

    Parameters:
        seg: EEG signal segment
        fs: Sampling frequency
        method: Artifact detection method ('amplitude', 'zscore', 'gamma_power')
        amp_thresh: Amplitude threshold for 'amplitude' method (μV)
        z_thresh: Z-score threshold for 'zscore' method
        z_outlier_fraction: Max fraction of samples allowed beyond threshold (default 5%)
        gamma_power_thresh: Gamma band power threshold for 'gamma_power' method
        gamma_band: Frequency range for gamma (low, high) in Hz


    Returns:
        bool: True if segment is clean, False if artifact detected
    """
    if method == "amplitude":
        return np.max(np.abs(seg)) > amp_thresh
    elif method == "zscore":
        z_scores = zscore(seg)
        max_allowed = int(len(seg) * z_outlier_fraction)
        outlier_count = np.sum(np.abs(z_scores) >= z_thresh)
        return outlier_count >= max_allowed
    elif method == "gamma_power":
        if gamma_power_thresh is None:
            raise ValueError("gamma_power_thresh must be provided")

        # Compute PSD with proper parameters
        nperseg = min(256, len(seg))
        freqs, psd = welch(
            seg, fs=fs, nperseg=nperseg, scaling="spectrum", detrend=False
        )

        # Calculate power metrics
        total_power = np.sum(psd)
        gamma_mask = (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])
        gamma_power = np.sum(psd[gamma_mask])

        # Combined relative + absolute power check
        if min_gamma_power is not None and gamma_power < min_gamma_power:
            return True  # Below absolute power threshold

        rel_gamma_power = gamma_power / total_power if total_power > 0 else 0
        return rel_gamma_power >= gamma_power_thresh
    else:
        raise ValueError(f"Unknown method '{method}'")


def detect_artifacts_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.Series:
    """
    Detect artifacts using the IQR method.

    Args:
        df: Input DataFrame.
        column: Column to apply detection on (e.g., 'max_abs_amplitude').
        k: IQR multiplier (default 1.5).

    Returns:
        Boolean Series: True if row is clean, False if artifact.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return (df[column] >= lower_bound) & (df[column] <= upper_bound)


def detect_artifacts_zscore(
    df: pd.DataFrame, column: str, threshold: float = 3.0
) -> pd.Series:
    """
    Detect artifacts using z-score method.

    Args:
        df: Input DataFrame.
        column: Column to apply detection on (e.g., 'gamma').
        threshold: Z-score threshold (default 3.0).

    Returns:
        Boolean Series: True if row is clean, False if artifact.
    """
    z = zscore(df[column])
    return np.abs(z) < threshold


def apply_artifact_rejection(
    segments: list[np.ndarray],
    fs: int = 256,
    amplitude_thresh: float = 100.0,
    zscore_thresh: float = 5.0,
    gamma_power_thresh: float = None,
) -> list[bool]:
    """
    Applies artifact detection to a list of segments.

    Returns:
    --------
    List of bools indicating which segments are clean (True = keep).
    """
    return [
        not detect_artifacts(
            seg, fs, amplitude_thresh, zscore_thresh, gamma_power_thresh
        )
        for seg in segments
    ]
