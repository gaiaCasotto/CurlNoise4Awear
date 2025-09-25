# src/awear_neuro/signal_processing/filters.py

from typing import Sequence

import numpy as np
import scipy.signal as ss


def bandpass_filter(
    x: Sequence[float],
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 47.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth band-pass filter to EEG data.

    Args:
        x: 1D signal array.
        fs: Sampling rate in Hz.
        lowcut: High-pass cutoff (default 0.5 Hz to remove drifts).
        highcut: Low-pass cutoff (default 47 Hz to retain EEG bands).
        order: Filter order (4th order gives 8th order zero-phase).

    Returns:
        Filtered signal as numpy array.
    """
    nyq = fs / 2.0
    b, a = ss.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return ss.filtfilt(b, a, x)


def notch_filter(
    x: Sequence[float],
    fs: float,
    freq: float = 60.0,
    Q: float = 30.0,
) -> np.ndarray:
    """
    Apply zero-phase IIR notch filter at specified power-line frequency.

    Args:
        x: 1D signal array.
        fs: Sampling rate in Hz.
        freq: Center of notch filter (commonly 50 or 60 Hz, depends on the country).
        Q: Quality factor controlling notch bandwidth.

    Returns:
        Filtered signal as numpy array.
    """
    nyq = fs / 2.0
    w0 = freq / nyq
    b, a = ss.iirnotch(w0, Q)
    #  zero-phase filtering with filtfilt() avoids phase distortions
    return ss.filtfilt(b, a, x)


def preprocess_segment(x: Sequence[float], fs: float) -> np.ndarray:
    """
    Preprocess a raw EEG segment: remove slow drifts, notch line noise, and detrend.

    Args:
        x: Input 1D signal array.
        fs: Sampling frequency in Hz.

    Returns:
        Preprocessed signal array.
    """
    x = bandpass_filter(x, fs)
    x = notch_filter(x, fs)
    # Detrending after filtering removes residual DC offset efficiently
    return ss.detrend(x)
