from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from awear_neuroscience.data_extraction.firestore_loader import (
    process_eeg_records, query_eeg_data)
from awear_neuroscience.data_extraction.reshape import (construct_long_df,
                                                        normalize_session)
from awear_neuroscience.signal_processing.artifacts import detect_artifacts
from awear_neuroscience.signal_processing.features import (
    add_time_features, apply_ema_filtering, compute_psd, extract_band_features,
    normalize_indexes)
from awear_neuroscience.signal_processing.filters import preprocess_segment


def process_long_df(
    long_df: pd.DataFrame,
    sampling_rate: int,
    artifacts_detection_method: str = "amplitude",
    amplitude_threshold: float = 20,
    **artifact_kwargs
) -> pd.DataFrame:
    """
    Process the “long” DataFrame: segment-wise filtering,
    max-abs annotation, and artifact‐flagging.

    Parameters
    ----------
    long_df : DataFrame
        Input EEG long-format DataFrame.
    sampling_rate : int
        Fs for both preprocess_segment and detect_artifacts.
    method : str, default 'amplitude'
        Artifact detection method.
    amplitude_threshold : float, default 20
        Amplitude threshold (used if method='amplitude').
    **artifact_kwargs :
        Extra method-specific kwargs for detect_artifacts.

    Returns
    -------
    long_df : pd.DataFrame
        With columns ['filtered_value','abs_filtered','max_abs_filtered_value','is_artifact',…].
    """

    # 1) segment-wise filtering
    segments = long_df["segment"].unique()
    filtered = [
        preprocess_segment(
            long_df.loc[long_df.segment == seg, "waveform_value"].values, sampling_rate
        )
        for seg in segments
    ]
    long_df["filtered_value"] = np.concatenate(filtered)
    long_df["abs_filtered"] = np.abs(long_df["filtered_value"])

    # 2) max-abs annotation
    max_abs = (
        long_df.groupby("segment")["abs_filtered"]
        .max()
        .reset_index(name="max_abs_filtered_value")
    )
    long_df = long_df.merge(max_abs, on="segment", how="left")

    # 3) cleanup
    long_df = long_df.dropna(subset=["segment", "filtered_value"]).reset_index(
        drop=True
    )

    # 4) artifact detection
    flags = (
        long_df.groupby("segment")["filtered_value"]
        .apply(
            lambda x: detect_artifacts(
                x.values,
                fs=sampling_rate,
                method=artifacts_detection_method,
                amp_thresh=amplitude_threshold,
                **artifact_kwargs
            )
        )
        .reset_index(name="is_artifact")
    )
    long_df = long_df.merge(flags, on="segment", how="left")

    return long_df


def extract_features_from_long_df(
    long_df: pd.DataFrame, sampling_rate: int
) -> pd.DataFrame:
    """
    For each non‐artifact segment in long_df, compute PSD and extract band features,
    preserving optional document_name and session_id in the output.

    Parameters
    ----------
    long_df : pd.DataFrame
        Must include columns:
          - 'segment', 'filtered_value', 'is_artifact',
          - 'focus_type', 'timestamp'
        May also include:
          - 'document_name', 'session_id'
    sampling_rate : float
        Fs for compute_psd.

    Returns
    -------
    features_df : pd.DataFrame
        One row per non‐artifact segment, with all band features plus:
          - segment
          - focus_type
          - timestamp
          - (optional) document_name
          - (optional) session_id
    """
    features = []
    for seg, seg_df in long_df.groupby("segment"):
        if not seg_df["is_artifact"].iloc[0]:
            signal = seg_df["filtered_value"].values
            freqs, psd = compute_psd(signal, sampling_rate)

            # Base feature dict
            feat = extract_band_features(
                freqs,
                psd,
                segment=seg,
                focus_type=seg_df["focus_type"].iloc[0],
                timestamp=seg_df["timestamp"].iloc[0],
            )

            # Inject optional metadata if it exists
            if "document_name" in seg_df.columns:
                feat["document_name"] = seg_df["document_name"].iloc[0]
            if "session_id" in seg_df.columns:
                feat["session_id"] = seg_df["session_id"].iloc[0]

            features.append(feat)

    return pd.DataFrame(features)


def process_features(
    features_df: pd.DataFrame,
    alpha: float,
    columns_to_normalize: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Sequentially apply EMA filtering, optional index normalization, and time feature engineering.

    Parameters
    ----------
    features_df : pd.DataFrame
        The input features DataFrame.
    alpha : float
        Smoothing factor for exponential moving average filtering.
    columns_to_normalize : List[str], optional
        Column names in `features_df` to be normalized after filtering.
        If None or empty, normalization is skipped.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with EMA filtering applied, specified columns normalized (if any),
        and additional time-based features added.
    """
    # 1) EMA smoothing
    df = apply_ema_filtering(features_df, alpha=alpha)

    # 2) Normalize selected columns if provided
    if columns_to_normalize:
        df = normalize_indexes(df, columns_to_normalize)

    # 3) Add derived time features
    df = add_time_features(df)

    return df
