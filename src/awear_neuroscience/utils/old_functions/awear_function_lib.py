import copy
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import signal, stats
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests

# ========================== #
# EEG Data Loading
# ========================== #


def load_eeg_data(file_paths, data_type, segments=None, fs=256, labels=None, channel=0):
    """
    Loads EEG data from multiple files and applies optional segmentation.

    Parameters:
        file_paths (list): List of EEG file paths.
        data_type (str): EEG data type ('awear' or 'openbci').
        segments (tuple, list, or None, optional):
            - **Single tuple** `(60,120)`: applies to all files.
            - **List of tuples** `[(60,120), None, (30,90), None]`: applies per file.
            - **None**: loads full data.
        fs (int): Sampling frequency.
        labels (list, optional): Custom labels for the datasets.
        channel (int, optional): EEG channel to extract for OpenBCI data (0,1,2, or 3). Default is **0**.

    Returns:
        dict: Dictionary of EEG data with dataset labels as keys.
    """
    if labels is None:
        labels = [
            os.path.basename(f).split(".")[0] for f in file_paths
        ]  # Default: use filename

    # Handle `segments` input properly
    if isinstance(segments, tuple):
        segments = [segments] * len(file_paths)  # Apply to all files
    elif isinstance(segments, list) and len(segments) == 1:
        segments = segments * len(file_paths)  # Expand single entry to match all files
    elif segments is None:
        segments = [None] * len(file_paths)  # No segmentation for all files

    # Ensure segment list length matches file_paths
    if len(segments) != len(file_paths):
        raise ValueError(
            "`segments` must be a tuple, None, or a list matching `file_paths` length."
        )

    # Validate OpenBCI channel selection
    if data_type == "openbci" and not (0 <= channel <= 3):
        raise ValueError("For OpenBCI data, `channel` must be between 0 and 3.")

    data_dict = {}

    for file_path, label, segment in zip(file_paths, labels, segments):
        if data_type == "awear":
            data_file = pd.read_csv(file_path, header=None, delimiter=",")
            data = data_file.iloc[:, 3].values  # Assume EEG data is in the 4th column
        elif data_type == "openbci":
            data_file = pd.read_csv(file_path, delimiter=",", skiprows=4)
            data = data_file.iloc[
                :, channel + 1
            ].values  # Shift index to skip first column
        else:
            raise ValueError("Invalid data_type. Must be 'openbci' or 'awear'.")

        # Apply segmentation if needed
        if segment:
            data = apply_segment(data, segment, fs)

        data_dict[label] = data

    return data_dict


def apply_segment(data, segment, fs):
    """
    Extracts a time-based segment from a 1D EEG signal.

    Parameters:
        data (np.ndarray): EEG signal as a 1D NumPy array.
        segment (tuple): Start and end times (in seconds) for the segment. Can include negative values.
        fs (int): Sampling frequency in Hz.

    Returns:
        np.ndarray: Segmented portion of the EEG signal.
    """
    # If no segment is provided, return the full signal
    if segment is None:
        return data

    start, end = segment

    # Convert time (in seconds) to sample indices
    # Allows negative indexing relative to the end of the signal
    start_sample = (
        max(int(start * fs), 0) if start >= 0 else max(len(data) + int(start * fs), 0)
    )
    end_sample = max(int(end * fs), 0) if end > 0 else max(len(data) + int(end * fs), 0)

    # Sanity check: start index must be less than end index
    if start_sample >= end_sample:
        raise ValueError(f"Invalid segment: start={start}, end={end}")

    # Return the sliced portion of the EEG signal
    return data[start_sample:end_sample]


# Function for removing outliers
def remove_outliers(data_dict, threshold=2):
    """
    Removes outliers from EEG data using a threshold-based approach and linear interpolation.

    Parameters:
        data_dict (dict): Dictionary containing EEG data with states as keys.
        threshold (float, optional): Number of standard deviations from the mean to define outliers. Default is 2.

    Returns:
        dict: Dictionary with cleaned EEG data.
    """
    cleaned_data = {}

    for state, data in data_dict.items():
        # Identify outliers
        mask = abs(data - np.mean(data)) > threshold * np.std(data)

        # Replace outliers using linear interpolation
        indices = np.arange(len(data))
        cleaned = np.copy(data)
        cleaned[mask] = np.interp(indices[mask], indices[~mask], data[~mask])

        cleaned_data[state] = cleaned

    return cleaned_data


# ========================== #
# EEG Filtering
# ========================== #
def apply_filters(
    data, fs, bandpass_range=None, notch_frequencies=None, bandpass_order=None, Q=None
):
    """
    Applies bandpass and notch filters to EEG data.

    Parameters:
        data (numpy.ndarray or dict): EEG signal or a dictionary of signals per condition.
        fs (int): Sampling frequency in Hz.
        bandpass_range (tuple, optional): Bandpass frequency range (low, high). Default: (0.5, 54).
        notch_frequencies (list, optional): List of frequencies to apply notch filters at. Default: [50, 60].
        bandpass_order (int, optional): Order of the bandpass filter. Default: 4.
        Q (int, optional): Quality factor for notch filter. Default: 10.

    Returns:
        dict or numpy.ndarray: Filtered EEG data (same structure as input).
    """
    # If input is a dictionary of signals, apply filters recursively to each entry
    if isinstance(data, dict):
        return {
            label: apply_filters(
                d, fs, bandpass_range, notch_frequencies, bandpass_order, Q
            )
            for label, d in data.items()
        }

    # Set default parameters if not provided
    if bandpass_range is None:
        bandpass_range = (0.5, 54)
    if notch_frequencies is None:
        notch_frequencies = [50, 60]
    if bandpass_order is None:
        bandpass_order = 4
    if Q is None:
        Q = 10

    # Apply bandpass filter
    lowcut, highcut = bandpass_range
    data = butter_bandpass_filter(data, lowcut, highcut, fs, order=bandpass_order)

    # Apply notch filter at each specified frequency
    for freq in notch_frequencies:
        data = notch_filter(data, freq, fs, Q=Q)

    return data


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a Butterworth bandpass filter to isolate a frequency range.

    Parameters:
        data (np.ndarray): 1D EEG signal.
        lowcut (float): Low frequency cutoff (Hz).
        highcut (float): High frequency cutoff (Hz).
        fs (int): Sampling frequency (Hz).
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(
        b, a, data
    )  # Apply forward-backward filter to avoid phase distortion


def notch_filter(data, freq, fs, Q=10):
    """
    Applies a notch filter to remove a specific frequency (e.g., power line noise).

    Parameters:
        data (np.ndarray): 1D EEG signal.
        freq (float): Frequency to filter out (e.g., 50 or 60 Hz).
        fs (int): Sampling frequency (Hz).
        Q (float): Quality factor — determines filter width.

    Returns:
        np.ndarray: Filtered signal with the target frequency attenuated.
    """
    nyquist = 0.5 * fs
    w0 = freq / nyquist  # Normalize frequency
    b, a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, data)  # Apply zero-phase filtering


# ========================== #
# Feature Extraction
# ========================== #

frequency_bands = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "alpha1": [8, 10],
    "alpha2": [10, 12],
    "alpha": ["alpha1", "alpha2"],  # Average of alpha1 & alpha2
    "beta1": [12, 18],
    "beta2": [18, 24],
    "beta3": [24, 30],
    "beta": ["beta1", "beta2", "beta3"],  # Average of beta1, beta2, beta3
    "gamma1": [30, 38],
    "gamma2": [38, 46],
    "gamma": ["gamma1", "gamma2"],  # Average of gamma1 & gamma2
}


def compute_band_power(freqs, psd, band):
    """
    Computes the total power within a specified frequency band from a PSD estimate.
    Handles both atomic (e.g., 'alpha1') and grouped bands (e.g., 'alpha').

    Parameters:
        freqs (np.ndarray): Array of frequency bins from the PSD.
        psd (np.ndarray): Power spectral density values corresponding to 'freqs'.
        band (str): Name of the frequency band to compute power for.

    Returns:
        float: Total power within the specified frequency band.
    """
    # If the band is a grouped label (like 'alpha' or 'beta'), compute power for each sub-band
    if isinstance(frequency_bands[band][0], str):
        sub_band_powers = []
        for sub_band in frequency_bands[band]:
            f_low, f_high = frequency_bands[sub_band]
            # Sum power in the sub-band range
            power = np.sum(psd[(freqs >= f_low) & (freqs <= f_high)])
            sub_band_powers.append(power)
        # Return average power across sub-bands
        return np.mean(sub_band_powers)
    else:
        # If it's an atomic band (like 'theta' or 'gamma1'), directly compute total power
        f_low, f_high = frequency_bands[band]
        return np.sum(psd[(freqs >= f_low) & (freqs <= f_high)])


def calculate_ratios(data_dict, fs, selected_ratios):
    """
    Calculates power ratios between selected frequency bands for each dataset.

    Parameters:
        data_dict (dict): EEG signal dictionary, with keys as condition labels and values as 1D EEG arrays.
        fs (int): Sampling frequency in Hz.
        selected_ratios (list of tuples): List of band pairs to compute ratios from,
                                          e.g., [('gamma2', 'alpha'), ('beta', 'delta')].

    Returns:
        dict: Nested dictionary in the form {condition: {ratio_label: ratio_value}}.
    """
    # Initialize a dictionary to hold the results for each dataset
    ratios = {file: {} for file in data_dict.keys()}

    for file, data in data_dict.items():
        # Estimate power spectral density using Welch's method
        freqs, psd = signal.welch(
            data, fs, nperseg=int(fs), noverlap=int(fs * 0.50), window="hann"
        )

        for band1, band2 in selected_ratios:
            # Compute absolute power in each selected band
            band1_power = compute_band_power(freqs, psd, band1)
            band2_power = compute_band_power(freqs, psd, band2)

            # Avoid division by zero by setting a small floor value
            ratio = band1_power / band2_power if band2_power != 0 else 1e-6

            # Store result using a readable label like "gamma2/alpha"
            ratios[file][f"{band1}/{band2}"] = ratio

    return ratios


def apply_lzc_to_data(data_dict):
    """
    Computes Lempel-Ziv Complexity (LZC) for each dataset.

    Parameters:
        data_dict (dict): Dictionary containing EEG datasets.

    Returns:
        dict: Dictionary of LZC values for each dataset.
    """

    def calculate_lzc(data):
        # Convert signal into a binary sequence using the median as the threshold
        binary_seq = (data > np.median(data)).astype(int)

        # Turn binary sequence into a string for parsing
        binary_str = "".join(binary_seq.astype(str))

        # Initialize variables for LZC parsing
        n = len(binary_str)
        i, k = 0, 1
        lzc_count = 1  # At least one distinct pattern exists by default

        # Parse through the string looking for new subsequences
        while i + k < n:
            # If the current pattern already exists earlier in the string
            if binary_str[: i + k].find(binary_str[i : i + k + 1]) != -1:
                k += 1
            else:
                # Found a new subsequence, increment count and move window
                lzc_count += 1
                i += k
                k = 1

        # Normalize complexity value by sequence length
        norm_factor = n / np.log2(n) if n > 1 else 1
        return lzc_count / norm_factor

    # Compute LZC for each dataset in the input dictionary
    return {file: calculate_lzc(data) for file, data in data_dict.items()}


# Phase-Amplitude Coupling Function


def compute_time_resolved_pac(
    data,
    fs=200,
    window_length=1,
    overlap=0.5,
    freq_range_phase=(8, 12),
    freq_range_amplitude=(30, 50),
    num_bins=18,
    smooth_window=8,
    zscore_threshold=2,
):
    """
    Compute time-resolved Phase-Amplitude Coupling (PAC) with Z-score normalization and outlier removal.

    Parameters:
        data (dict): Dictionary containing raw EEG data for each state.
        fs (int): Sampling frequency (default: 200 Hz).
        window_length (float): Time window length in seconds (default: 1s).
        overlap (float): Overlap fraction between windows (default: 0.5 for 50% overlap).
        freq_range_phase (tuple): Band for phase extraction (default: 8-12 Hz).
        freq_range_amplitude (tuple): Band for amplitude extraction (default: 30-50 Hz).
        num_bins (int): Number of phase bins for computing PAC (default: 18).
        smooth_window (int): Window size for moving average smoothing (default: 8).
        zscore_threshold (float): Threshold for removing extreme PAC values (default: 2).

    Returns:
        smoothed_pac_series (dict): Cleaned & smoothed PAC time series per state.
    """
    # Convert window length and step size to samples
    win_samples = int(window_length * fs)
    step_size = int(win_samples * (1 - overlap))

    pac_time_series = {state: [] for state in data.keys()}

    # Compute PAC for each time window
    for state, signal_data in data.items():
        num_samples = len(signal_data)

        for start in range(0, num_samples - win_samples, step_size):
            window_data = signal_data[start : start + win_samples]

            # Apply filters using our existing function
            theta_filtered = apply_filters(
                window_data, bandpass_range=freq_range_phase, fs=fs
            )
            gamma_filtered = apply_filters(
                window_data, bandpass_range=freq_range_amplitude, fs=fs
            )

            # Compute phase and amplitude
            theta_phase = np.angle(signal.hilbert(theta_filtered))
            gamma_amplitude = np.abs(signal.hilbert(gamma_filtered))

            # Compute PAC for the window
            mi_value = compute_pac(theta_phase, gamma_amplitude, num_bins)
            pac_time_series[state].append(mi_value)

    # Convert PAC time series to DataFrame for Z-score normalization
    pac_df_raw = pd.DataFrame.from_dict(pac_time_series, orient="index").T

    # Apply Z-score normalization & remove extreme values
    pac_zscored = (pac_df_raw - pac_df_raw.mean()) / pac_df_raw.std()
    pac_df_cleaned = pac_df_raw.mask(pac_zscored.abs() > zscore_threshold).dropna()

    # Apply smoothing to PAC time series
    smoothed_pac_series = {
        state: moving_average(values.dropna(), smooth_window)
        for state, values in pac_df_cleaned.items()
    }

    # Plot PAC time series
    plt.figure(figsize=(10, 5))
    for state, pac_values in smoothed_pac_series.items():
        plt.plot(pac_values, label=state, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("PAC Strength (MI)")
    plt.title(
        f"Time-Resolved PAC ({freq_range_phase[0]}-{freq_range_phase[1]} Hz Phase & {freq_range_amplitude[0]}-{freq_range_amplitude[1]} Hz Amplitude)"
    )
    plt.legend()
    plt.show()

    return smoothed_pac_series


def compute_pac(theta_phase, gamma_amplitude, num_bins):
    """Computes PAC using KL divergence based on gamma amplitude binned by theta phase."""

    # Divide the -π to π range into bins for phase values
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    bin_indices = (
        np.digitize(theta_phase, bin_edges) - 1
    )  # Assign each phase value to a bin

    # For each bin, compute the mean amplitude of the gamma signal where the phase falls into that bin
    mean_amplitude = np.array(
        [
            gamma_amplitude[bin_indices == i].mean() if np.any(bin_indices == i) else 0
            for i in range(num_bins)
        ]
    )

    # Normalize to form a probability distribution, ensuring no division by zero
    mean_amplitude = np.where(
        mean_amplitude > 0, mean_amplitude / mean_amplitude.sum(), 1e-10
    )

    # Compare the observed amplitude distribution to a uniform distribution
    # If the distribution is non-uniform, it suggests phase-amplitude coupling
    uniform_dist = np.ones(num_bins) / num_bins
    return stats.entropy(mean_amplitude, uniform_dist)


def moving_average(data, window_size):
    """Smooths a 1D time series using a moving average filter."""
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# Compute effect sizes for PAC


def compute_effect_sizes(pac_data, name):
    """
    Computes Cohen's d, rank-biserial correlation (r), and Mann-Whitney U test for PAC comparisons.

    Parameters:
    - pac_data: Dictionary containing PAC values for each emotional state.
    - name: Name of the participant for identification.

    Returns:
    - DataFrame containing effect sizes and statistical results for each state comparison.
    """

    state_pairs = list(itertools.combinations(pac_data.keys(), 2))
    results = []

    for state1, state2 in state_pairs:
        data1 = pac_data[state1]
        data2 = pac_data[state2]

        # Skip empty or too-small datasets
        if len(data1) < 2 or len(data2) < 2:
            continue

        # Compute Cohen’s d
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

        if std1 == 0 or std2 == 0:  # Avoid division by zero
            cohen_d = np.nan
        else:
            pooled_std = np.sqrt(
                ((len(data1) - 1) * std1**2 + (len(data2) - 1) * std2**2)
                / (len(data1) + len(data2) - 2)
            )
            cohen_d = (mean1 - mean2) / pooled_std

        # Compute Rank-Biserial Correlation (r) and Mann-Whitney U test
        U, p = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        r = 1 - (2 * U) / (len(data1) * len(data2))

        results.append(
            {
                "Comparison": f"{state1} vs {state2}",
                "Cohen's d": cohen_d,
                "Rank-Biserial r": r,
                "Mann-Whitney p-value": p,
            }
        )

    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    result_df.reset_index(drop=True, inplace=True)

    # Suppress automatic Jupyter output by returning None
    print(f"\n{name.capitalize()} Statistical Results:\n")
    display(result_df)  # Ensures clean display in Jupyter

    return None


# Sample Entropy Computation


def compute_sample_entropy(data_dict, fs, window_size=1000, overlap=0.5, m=2, r=0.2):
    """
    Computes Sample Entropy (SampEn) using a sliding window across EEG states.

    Parameters:
    - data_dict (dict): EEG data dictionary where keys are state labels and values are EEG signals.
    - fs (int): Sampling frequency.
    - window_size (int): Number of samples per window (default: 1000).
    - overlap (float): Fraction of overlap between consecutive windows (default: 0.5).
    - m (int): Embedding dimension for Sample Entropy calculation (default: 2).
    - r (float): Tolerance as a fraction of signal standard deviation (default: 0.2).

    Returns:
    - DataFrame: Sample Entropy values for each state and time window.
    """

    def compute_sample_entropy(signal, m, r):
        """Computes Sample Entropy (SampEn) efficiently."""
        N = len(signal)
        r *= np.std(signal)  # Convert relative tolerance to absolute scale

        if N < m + 1 or np.std(signal) == 0:
            return np.nan  # Avoid division by zero issues

        def _phi(m):
            """Helper function to compute matching template counts."""
            templates = np.array([signal[i : i + m] for i in range(N - m)])
            dist_matrix = cdist(
                templates, templates, metric="chebyshev"
            )  # Use Euclidean or Chebyshev
            count = np.sum(dist_matrix < r, axis=1) - 1  # Remove self-matches
            return np.mean(count) / (N - m)

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        if phi_m == 0 or phi_m1 == 0:
            return np.nan  # Undefined entropy when division by zero occurs

        return -np.log(phi_m1 / phi_m)

    results = []
    step_size = int(window_size * (1 - overlap))  # Compute step size

    for state, signal in data_dict.items():
        if len(signal) < window_size:
            continue  # Skip states with insufficient data

        for start in range(0, len(signal) - window_size + 1, step_size):
            segment = signal[start : start + window_size]
            samp_en = compute_sample_entropy(segment, m, r)

            results.append(
                {
                    "State": state,
                    "Window Start (s)": start / fs,
                    "Window End (s)": (start + window_size) / fs,
                    "Sample Entropy": samp_en,
                }
            )

    return pd.DataFrame(results)


# Approximate Entropy Computation


def compute_approximate_entropy(
    data_dict, fs, window_size=1000, overlap=0.5, m=2, r=0.2
):
    """
    Computes Approximate Entropy (ApEn) for EEG signals using a sliding window approach.

    Parameters:
    - data_dict (dict): Dictionary where keys are labels (e.g., emotional states) and values are EEG signals.
    - fs (int): Sampling frequency.
    - window_size (int): Number of samples per window (default: 1000).
    - overlap (float): Overlap fraction between consecutive windows (default: 0.5).
    - m (int): Embedding dimension (default: 2).
    - r (float): Tolerance as a fraction of signal standard deviation (default: 0.2).

    Returns:
    - DataFrame: Approximate Entropy values for each state and window.
    """

    def approximate_entropy(signal, m, r):
        """Computes Approximate Entropy for a single time series."""
        N = len(signal)
        r *= np.std(signal)  # Convert tolerance to absolute scale

        def _phi(m):
            templates = np.array([signal[i : i + m] for i in range(N - m)])
            dist_matrix = cdist(
                templates, templates, metric="chebyshev"
            )  # Use Euclidean or Chebyshev
            count = np.sum(dist_matrix < r, axis=1) / (N - m)  # Normalize count
            return np.mean(count)

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        return np.log(phi_m) - np.log(phi_m1)

    results = []
    step_size = int(window_size * (1 - overlap))  # Compute step size

    for state, signal in data_dict.items():
        if len(signal) < window_size:  # Skip if too short
            continue

        # Apply sliding window
        for start in range(0, len(signal) - window_size + 1, step_size):
            segment = signal[start : start + window_size]
            ap_en = approximate_entropy(segment, m, r)

            results.append(
                {
                    "State": state,
                    "Window Start (s)": start / fs,
                    "Window End (s)": (start + window_size) / fs,
                    "Approximate Entropy": ap_en,
                }
            )

    return pd.DataFrame(results)


# Compute mean and standard error of Entropy Values


def compute_average_entropy(entropy_df, entropy_type):
    """
    Computes the average entropy and standard error of the mean (SEM)
    for each state. Supports both multi-participant and single-participant cases.

    Parameters:
    - entropy_df (DataFrame): DataFrame containing entropy values.
    - entropy_type (str): Type of entropy ('Sample' or 'Approximate').

    Returns:
    - DataFrame with average entropy and SEM per state.
    """
    if entropy_type not in ["Sample", "Approximate"]:
        raise ValueError("Invalid entropy_type. Choose 'Sample' or 'Approximate'.")

    entropy_col = f"{entropy_type} Entropy"

    # Ensure the entropy column exists
    if entropy_col not in entropy_df.columns:
        raise KeyError(f"Column '{entropy_col}' not found in the provided DataFrame.")

    # Check if 'Participant' column exists; otherwise, add a default one
    if "Participant" not in entropy_df.columns:
        entropy_df = entropy_df.copy()
        entropy_df["Participant"] = "Dataset"  # Default label

    # Group by participant and state
    grouped = entropy_df.groupby(["Participant", "State"])[entropy_col]

    # Compute mean and SEM
    result_df = grouped.agg(
        Mean_Entropy="mean",
        SEM_Entropy=lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
        if len(x) > 1
        else np.nan,
    ).reset_index()

    return result_df


# Compute ANOVA test on Entropy Data


def compute_anova_entropy(entropy_df, entropy_type):
    """
    Performs One-Way ANOVA on entropy values across emotional states for each participant.

    Parameters:
    - entropy_df (DataFrame): DataFrame containing 'Participant', 'State', and entropy values.
    - entropy_type (str): Type of entropy to analyze ('Sample' or 'Approximate').

    Returns:
    - DataFrame with ANOVA results per participant, including F-statistic and p-value.
    """
    if entropy_type not in ["Sample", "Approximate"]:
        raise ValueError("Invalid entropy_type. Choose 'Sample' or 'Approximate'.")

    entropy_col = f"{entropy_type} Entropy"

    # Ensure the column exists
    if entropy_col not in entropy_df.columns:
        raise KeyError(f"Column '{entropy_col}' not found in the provided DataFrame.")

    results = []

    for participant in entropy_df["Participant"].unique():
        subset = entropy_df[entropy_df["Participant"] == participant]

        # Group entropy values by state
        grouped = [
            state_group[entropy_col].dropna().values
            for _, state_group in subset.groupby("State")
        ]

        # Ensure at least two states have multiple values for ANOVA
        if len(grouped) > 1 and all(len(g) > 1 for g in grouped):
            F_stat, p_value = stats.f_oneway(*grouped)
            p_value = f"{p_value:.2e}"  # Convert p-value to scientific notation
        else:
            F_stat, p_value = None, None  # Not enough valid data for ANOVA

        results.append(
            {"Participant": participant, "F-statistic": F_stat, "p-value": p_value}
        )

    return pd.DataFrame(results)


# Perform pairwise t test
def compute_pairwise_tests(entropy_df, entropy_type):
    """
    Performs pairwise t-tests for entropy values between all states per participant.

    Parameters:
    - entropy_df (DataFrame): DataFrame containing 'Participant', 'State', and entropy values.
    - entropy_type (str): Type of entropy to analyze ('Sample' or 'Approximate').

    Returns:
    - DataFrame with pairwise t-test results (p-values) and Bonferroni-corrected p-values.
    """
    if entropy_type not in ["Sample", "Approximate"]:
        raise ValueError("Invalid entropy_type. Choose 'Sample' or 'Approximate'.")

    entropy_col = f"{entropy_type} Entropy"

    # Ensure the column exists
    if entropy_col not in entropy_df.columns:
        raise KeyError(f"Column '{entropy_col}' not found in the provided DataFrame.")

    results = []

    for participant in entropy_df["Participant"].unique():
        subset = entropy_df[entropy_df["Participant"] == participant]

        # Generate all unique state pairs for pairwise comparisons
        state_pairs = list(itertools.combinations(subset["State"].unique(), 2))
        p_values = []

        for state1, state2 in state_pairs:
            data1 = subset[subset["State"] == state1][entropy_col].dropna()
            data2 = subset[subset["State"] == state2][entropy_col].dropna()

            # Only perform test if both groups have more than one value
            if len(data1) > 1 and len(data2) > 1:
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                p_values.append(p_value)
                results.append(
                    {
                        "Participant": participant,
                        "Comparison": f"{state1} vs {state2}",
                        "p-value": p_value,  # Raw p-value stored for correction
                    }
                )
            else:
                results.append(
                    {
                        "Participant": participant,
                        "Comparison": f"{state1} vs {state2}",
                        "p-value": None,  # Not enough data for valid test
                        "Corrected p-value": None,
                    }
                )

        # Apply Bonferroni correction if there are valid p-values
        valid_p_values = [
            res["p-value"]
            for res in results
            if res["Participant"] == participant and res["p-value"] is not None
        ]

        if valid_p_values:
            corrected_p_values = multipletests(valid_p_values, method="bonferroni")[1]

            # Assign corrected p-values
            corrected_index = 0
            for res in results:
                if res["Participant"] == participant and res["p-value"] is not None:
                    res["Corrected p-value"] = corrected_p_values[corrected_index]
                    corrected_index += 1

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df
