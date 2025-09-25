import os

from awear_function_lib import *
from plot_utils import *

# ========================== #
# Full Pipeline Execution
# ========================== #


def run_pipeline(
    data_type,
    file_paths,
    fs=256,
    selected_ratios=[("gamma1", "delta"), ("beta", "alpha")],  # Default ratios
    labels=None,
    segments=None,
    y_axis_limits=None,
    stft_power_scale=None,
    plots=None,
):
    """
    Runs the full EEG analysis pipeline.

    Parameters:
        data_type (str): EEG data type ('awear' or 'openbci').
        file_paths (list): List of EEG file paths.
        fs (int): Sampling frequency in Hz.
        selected_ratios (list of tuples): Band pairs to compare, e.g., [("gamma2", "alpha"), ("beta", "delta")].
        labels (list, optional): Custom dataset labels. Default is None.
        segments (list, optional): Time segments for each dataset. Default is None.
        y_axis_limits (dict, optional): Y-axis limits for plots. Default is None.
        stft_power_scale (int, optional): Max scale for spectrogram plots. Default is None.
        plots (dict, optional): Dictionary specifying which plots to generate. Default is None.

    Returns:
        None: Displays plots based on specified options.
    """
    # Assign default labels if none are provided
    if labels is None:
        labels = [os.path.basename(f).split(".")[0] for f in file_paths]

    # Assign default segmentation (None) if not provided
    if segments is None:
        segments = [None] * len(file_paths)

    # Load raw EEG data
    raw_data = load_eeg_data(file_paths, data_type, segments, fs, labels=labels)

    # Apply filtering while preserving labels
    data_dict = {label: apply_filters(raw_data[label], fs) for label in labels}

    # Default plot settings if none are provided
    if plots is None:
        plots = {
            "psd": True,
            "bar_ratios": True,
            "time_frequency": True,
            "band_ratios_over_time": True,
            "box_plot": True,
            "lzc": True,
        }

    # Generate PSD plot
    if plots.get("psd", True):
        plot_psd(data_dict, fs, y_axis_limits.get("psd") if y_axis_limits else None)

    # Generate bar plot for band ratios
    if plots.get("bar_ratios", True):
        ratios = calculate_ratios(data_dict, fs, selected_ratios)
        plot_ratios_bar(
            ratios, y_axis_limits.get("bar_ratios") if y_axis_limits else None
        )

    # Generate spectrogram plots
    if plots.get("time_frequency", True):
        plot_spectrogram(data_dict, fs, vmax=stft_power_scale)

    # Generate time-evolving band ratio plots
    if plots.get("band_ratios_over_time", True):
        plot_band_ratios_spectrogram(
            data_dict,
            fs,
            selected_ratios,
            y_axis_limits.get("band_ratios_over_time") if y_axis_limits else None,
        )

    # Generate box-and-whisker plots for band ratios
    if plots.get("box_plot", True):
        plot_band_ratios_box_whisker(
            data_dict,
            fs,
            selected_ratios,
            y_axis_limits.get("box_plot") if y_axis_limits else None,
        )

    # Compute and visualize LZC values
    if plots.get("lzc", True):
        lzc_results = apply_lzc_to_data(data_dict)
        plot_lzc_values(lzc_results)

    plt.show()
