import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy import signal


def generate_plotly_colors(num_colors):
    """
    Generates a list of distinct Plotly-compatible colors.

    Parameters:
        num_colors (int): Number of colors needed.

    Returns:
        list: A list of color hex codes or names suitable for Plotly.
    """
    # Plotly's default qualitative palette
    base_colors = px.colors.qualitative.Plotly

    # Extend color list if more are needed by repeating
    return (
        base_colors[:num_colors]
        if num_colors <= len(base_colors)
        else [base_colors[i % len(base_colors)] for i in range(num_colors)]
    )


def plot_eeg_waveform(df: pd.DataFrame, segment_id: str = "seg_0") -> None:
    """
    Plot EEG waveform for a specific segment using Plotly.
    """
    segment_df = df[df["segment"] == segment_id]

    fig = px.line(
        segment_df,
        x="time_sample",
        y="waveform_value",
        title=f"EEG Waveform for {segment_id}",
        labels={"time_sample": "Time (s)", "waveform_value": "Amplitude (µV)"},
    )
    fig.show()


import matplotlib.pyplot as plt

def plot_eeg_waveform_matplotlib(df: pd.DataFrame, segment_id: str = "seg_0") -> None:
    """
    Plot EEG waveform for a specific segment using matplotlib.
    """
    segment_df = df[df["segment"] == segment_id]

    plt.figure(figsize=(10, 4))
    plt.plot(
        segment_df["time_sample"], 
        segment_df["waveform_value"], 
        color="blue", linewidth=1
    )
    plt.title(f"EEG Waveform for {segment_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_band_ratio_heatmap_plotly(data_dict, fs, name=None, channel=None):
    """
    Plots an interactive heatmap of meaningful band power ratios using Plotly.

    Parameters:
        data_dict (dict): Dictionary containing EEG datasets with labels.
        fs (int): Sampling frequency.
        name (str): Name of the subject for labeling the heatmap.
        channel (int, optional): EEG channel number. Default is None.

    Returns:
        None: Displays the interactive Plotly heatmap.
    """
    # Define frequency bands
    frequency_bands = {
        "delta": ["delta"],
        "theta": ["theta"],
        "alpha": ["alpha", "alpha1", "alpha2"],
        "beta": ["beta", "beta1", "beta2", "beta3"],
        "gamma": ["gamma", "gamma1", "gamma2"],
    }

    # Generate unique band ratios: higher / lower
    selected_ratios = []
    main_bands = list(frequency_bands.keys())
    for i in range(len(main_bands)):
        for j in range(i + 1, len(main_bands)):
            for high in frequency_bands[main_bands[j]]:
                for low in frequency_bands[main_bands[i]]:
                    selected_ratios.append((high, low))

    # Compute band ratios
    band_ratios = calculate_ratios(
        data_dict, fs, selected_ratios
    )  # Should return {label: {ratio: value}}

    # Convert to DataFrame
    df = pd.DataFrame(band_ratios).T  # rows: labels (states), cols: (high/low)

    # Optional labeling
    channel_label = f"Channel {channel}" if channel is not None else ""
    title = (
        f"{name}'s Band Power Ratios Heatmap - {channel_label}"
        if name
        else "Band Power Ratios Heatmap"
    )

    # Plotly Heatmap
    fig = px.imshow(
        df,
        labels=dict(x="Band Ratio (High / Low)", y="State", color="Ratio Value"),
        x=df.columns,
        y=df.index,
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
    )

    fig.show()


def plot_psd_plotly(data_dict, fs, name=None, y_axis_limits=None):
    """
    Plots the Power Spectral Density (PSD) of EEG signals using Plotly.

    Parameters:
        data_dict (dict): Dictionary of EEG datasets.
        fs (int): Sampling frequency in Hz.
        name (str): Subject/session name to annotate the plot.
        y_axis_limits (tuple, optional): Y-axis limits for log scale. Default is None.

    Returns:
        None: Displays an interactive PSD plot.
    """
    colors = generate_plotly_colors(
        len(data_dict)
    )  # Use Plotly-compatible color generator
    fig = go.Figure()

    for idx, (label, data) in enumerate(data_dict.items()):
        # Compute Welch PSD
        freqs, psd = signal.welch(
            data, fs, nperseg=int(fs * 3), noverlap=int(fs * 0.5), window="hann"
        )

        # Add trace to figure (use log scale for y)
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=psd,
                mode="lines",
                name=label,
                line=dict(color=colors[idx]),
                hovertemplate="Freq: %{x:.2f} Hz<br>Power: %{y:.2e}<extra>%{fullData.name}</extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{name} Power Spectral Density (PSD)"
        if name
        else "Power Spectral Density (PSD)",
        xaxis=dict(title="Frequency (Hz)", range=[0, 80]),
        yaxis=dict(title="Power", type="log", range=y_axis_limits),
        height=400,
        template="plotly_white",
        legend_title_text="Condition",
        margin=dict(l=60, r=30, t=60, b=40),
    )

    fig.show()


def plot_ratios_bar_plotly(ratios, name=None, y_axis_limits=None):
    """
    Plots an interactive grouped bar chart of band power ratios using Plotly.

    Parameters:
        ratios (dict): Dictionary in format {file: {ratio_label: value, ...}, ...}
        name (str, optional): Title of the plot.
        y_axis_limits (tuple, optional): Y-axis (log) limits.

    Returns:
        None: Displays interactive bar chart.
    """
    files = list(ratios.keys())
    ratio_labels = list(next(iter(ratios.values())).keys())
    colors = generate_plotly_colors(len(ratio_labels))

    # Prepare data for Plotly: create a long-format DataFrame
    data = []
    for file in files:
        for ratio in ratio_labels:
            data.append(
                {
                    "File": file,
                    "Ratio": ratio.replace("_", "/"),
                    "Value": ratios[file][ratio],
                }
            )
    df = pd.DataFrame(data)

    # Create figure
    fig = go.Figure()

    for idx, ratio in enumerate(ratio_labels):
        filtered_df = df[df["Ratio"] == ratio.replace("_", "/")]
        fig.add_trace(
            go.Bar(
                x=filtered_df["File"],
                y=filtered_df["Value"],
                name=ratio.replace("_", "/"),
                marker_color=colors[idx],
            )
        )

    # Update layout
    fig.update_layout(
        barmode="group",
        title=f"{name} Frequency Band Ratios Comparison"
        if name
        else "Frequency Band Ratios Comparison",
        xaxis_title="File",
        yaxis_title="Ratio (Log Scale)",
        yaxis_type="log",
        yaxis_range=y_axis_limits,  # This is optional; only works if given as log10 scale range (e.g., [0, 2])
        legend_title_text="Band Ratio",
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.show()


def plot_spectrogram_plotly(data_dict, fs, name=None, vmax=None):
    """
    Plots interactive spectrograms for EEG signals using Plotly.

    Parameters:
        data_dict (dict): Dictionary of EEG datasets.
        fs (int): Sampling frequency in Hz.
        name (str, optional): Title of the plot.
        vmax (float, optional): Maximum power value for color intensity.

    Returns:
        None: Displays interactive spectrograms.
    """
    num_datasets = len(data_dict)
    fig = sp.make_subplots(
        rows=num_datasets,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        subplot_titles=[key.capitalize() for key in data_dict.keys()],
    )

    for idx, (label, data) in enumerate(data_dict.items(), start=1):
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(
            data, fs, nperseg=int(fs), noverlap=int(fs * 0.5), window="hann"
        )

        # Clip for consistent color range if vmax is given
        Sxx_display = np.clip(Sxx, 0, vmax) if vmax else Sxx

        fig.add_trace(
            go.Heatmap(
                z=Sxx_display,
                x=t,
                y=f,
                colorscale="Viridis",
                zmax=vmax,
                zmin=0,
                colorbar=dict(title="Power") if idx == 1 else None,
                showscale=(idx == 1),
            ),
            row=idx,
            col=1,
        )

        # Y-axis settings
        fig.update_yaxes(title_text="Freq (Hz)", range=[0, 50], row=idx, col=1)
        fig.update_xaxes(title_text="Time (s)", row=idx, col=1)

    # Layout and title
    fig.update_layout(
        height=300 * num_datasets,
        title_text=f"{name} Time-Frequency Domain Plots"
        if name
        else "Time-Frequency Domain Plots",
        template="plotly_white",
        margin=dict(t=60, l=60, r=30, b=40),
    )

    fig.show()


def plot_band_ratios_spectrogram_plotly(
    data_dict, fs, selected_ratios, frequency_bands, name=None, y_axis_limits=None
):
    """
    Plots the time evolution of selected band ratios using Plotly.

    Parameters:
        data_dict (dict): EEG datasets keyed by condition name.
        fs (int): Sampling frequency in Hz.
        selected_ratios (list of tuples): Band ratio pairs like [("gamma2", "alpha"), ("beta", "delta")].
        frequency_bands (dict): Mapping of band names to (f_min, f_max) tuples or composite sub-band lists.
        name (str, optional): Plot title.
        y_axis_limits (tuple, optional): Y-axis range for log scale.

    Returns:
        None
    """
    colors = generate_plotly_colors(len(data_dict))
    num_ratios = len(selected_ratios)

    fig = sp.make_subplots(
        rows=num_ratios,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        subplot_titles=[f"{b1}/{b2}".capitalize() for (b1, b2) in selected_ratios],
    )

    for ratio_idx, (band1, band2) in enumerate(selected_ratios):
        for file_idx, (label, data) in enumerate(data_dict.items()):
            # Compute spectrogram
            f, t, spg = signal.spectrogram(
                data, fs, nperseg=int(fs), noverlap=int(fs * 0.5), window="hann"
            )

            def get_band_power(band):
                # Recursive resolution for composite bands
                if isinstance(frequency_bands[band], list):
                    return np.mean(
                        [get_band_power(sub) for sub in frequency_bands[band]], axis=0
                    )
                else:
                    f_min, f_max = frequency_bands[band]
                    return spg[np.logical_and(f >= f_min, f <= f_max), :].mean(axis=0)

            # Compute band powers
            power_band1 = get_band_power(band1)
            power_band2 = get_band_power(band2)
            ratio = np.maximum(power_band1 / power_band2, 1e-6)

            # Plot
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=ratio,
                    mode="lines",
                    name=label if ratio_idx == 0 else None,
                    legendgroup=label,
                    showlegend=(ratio_idx == 0),
                    line=dict(color=colors[file_idx]),
                    hovertemplate="Time: %{x:.2f}s<br>Ratio: %{y:.2e}<extra>%{fullData.name}</extra>",
                ),
                row=ratio_idx + 1,
                col=1,
            )

        fig.update_yaxes(
            title_text="Ratio (log)",
            type="log",
            range=y_axis_limits,
            row=ratio_idx + 1,
            col=1,
        )
        fig.update_xaxes(title_text="Time (s)", row=ratio_idx + 1, col=1)

    fig.update_layout(
        height=300 * num_ratios,
        title=f"{name} Band Ratios Over Time" if name else "Band Ratios Over Time",
        template="plotly_white",
        legend_title="Condition",
        margin=dict(l=60, r=30, t=60, b=40),
    )

    fig.show()


def plot_band_ratios_box_whisker_plotly(
    data_dict, fs, selected_ratios, frequency_bands, name=None, y_axis_limits=None
):
    """
    Interactive box-and-whisker plot for comparing EEG band ratios using Plotly.

    Parameters:
        data_dict (dict): Dictionary of EEG data arrays.
        fs (int): Sampling frequency in Hz.
        selected_ratios (list): List of (high_band, low_band) tuples.
        frequency_bands (dict): Mapping of band names to (f_min, f_max) or sub-band lists.
        name (str): Optional title for the figure.
        y_axis_limits (tuple): Optional log-scale y-axis limits.

    Returns:
        None
    """
    colors = generate_plotly_colors(len(data_dict))
    fig = go.Figure()

    for ratio_idx, (band1, band2) in enumerate(selected_ratios):
        for file_idx, (file, data) in enumerate(data_dict.items()):
            # Compute spectrogram
            f, t, spg = signal.spectrogram(
                data, fs, nperseg=int(fs), noverlap=int(fs * 0.5), window="hann"
            )

            def get_band_power(band):
                if isinstance(frequency_bands[band], list):
                    return np.mean(
                        [
                            get_band_power(sub_band)
                            for sub_band in frequency_bands[band]
                        ],
                        axis=0,
                    )
                else:
                    f_min, f_max = frequency_bands[band]
                    return spg[np.logical_and(f >= f_min, f <= f_max)].mean(axis=0)

            power_band1 = get_band_power(band1)
            power_band2 = get_band_power(band2)
            ratio_values = np.maximum(power_band1 / power_band2, 1e-6)

            fig.add_trace(
                go.Box(
                    y=ratio_values,
                    name=f"{band1}/{band2}<br>{file.capitalize()}",
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color=colors[file_idx],
                    line_color=colors[file_idx],
                    marker=dict(size=4, opacity=0.6),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"{name} Band Ratios Box and Whisker Plot"
        if name
        else "Band Ratios Box and Whisker Plot",
        yaxis_title="Ratio (log scale)",
        yaxis_type="log",
        yaxis_range=np.log10(y_axis_limits) if y_axis_limits else None,
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
        height=500,
    )

    fig.show()


def plot_lzc_values_plotly(lzc_results):
    """
    Interactive bar chart of Lempel-Ziv Complexity (LZC) values using Plotly.

    Parameters:
        lzc_results (dict): Mapping of dataset label -> LZC value (0 to 1).

    Returns:
        None
    """
    activities = list(lzc_results.keys())
    lzc_values = list(lzc_results.values())
    colors = generate_plotly_colors(len(activities))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=activities,
            y=lzc_values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in lzc_values],
            textposition="outside",
            textfont=dict(size=12, color="black"),
            hovertemplate="Activity: %{x}<br>LZC: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="LZC Values for Different Activities",
        yaxis_title="LZC Value",
        yaxis=dict(range=[0, 1]),
        xaxis_title="Activity",
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=40, t=60, b=40),
    )

    fig.show()


def plot_avg_power_heatmap_plotly(
    data_dict, fs, compute_band_power, name=None, channel=None
):
    """
    Interactive heatmap of average EEG power across frequency bands and states using Plotly.

    Parameters:
        data_dict (dict): EEG data by condition label.
        fs (int): Sampling frequency in Hz.
        compute_band_power (func): Function to compute band power per (freqs, psd, band_name).
        name (str): Optional subject/session label.
        channel (int): Optional channel label.

    Returns:
        None
    """
    # Define all bands to include
    frequency_bands = [
        "delta",
        "theta",
        "alpha",
        "alpha1",
        "alpha2",
        "beta",
        "beta1",
        "beta2",
        "beta3",
        "gamma",
        "gamma1",
        "gamma2",
    ]

    # Compute average power for each condition and band
    avg_power = {}
    for label, data in data_dict.items():
        freqs, psd = signal.welch(data, fs, nperseg=int(fs))
        avg_power[label] = {
            band: compute_band_power(freqs, psd, band) for band in frequency_bands
        }

    # Create DataFrame and reshape for Plotly
    df = pd.DataFrame(avg_power).T  # rows = states, columns = bands
    df_reset = df.reset_index().melt(
        id_vars="index", var_name="Band", value_name="Power"
    )
    df_reset.rename(columns={"index": "State"}, inplace=True)

    # Plotly heatmap
    fig = px.imshow(
        df.values,
        x=df.columns,
        y=df.index,
        labels=dict(x="Frequency Band", y="State", color="Power"),
        color_continuous_scale="Viridis",
        text_auto=".2f",
    )

    # Title formatting
    title_channel = f" - Channel {channel}" if channel is not None else ""
    fig.update_layout(
        title=f"{name}'s Average Power Across Frequency Bands and States{title_channel}"
        if name
        else "Average Power Across Frequency Bands and States",
        xaxis_title="Frequency Bands",
        yaxis_title="States",
        xaxis_tickangle=45,
        height=500,
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.show()


def plot_band_ratio_heatmap_plotly(
    data_dict, fs, calculate_ratios, name=None, channel=None
):
    """
    Plots an interactive heatmap of meaningful EEG band power ratios using Plotly.

    Parameters:
        data_dict (dict): EEG datasets keyed by condition/label.
        fs (int): Sampling frequency.
        calculate_ratios (func): Function that computes {state: {ratio: value}} given data_dict, fs, and selected_ratios.
        name (str): Subject/session label for the plot.
        channel (int, optional): Channel identifier.

    Returns:
        None
    """
    # Define hierarchical frequency bands
    frequency_bands = {
        "delta": ["delta"],
        "theta": ["theta"],
        "alpha": ["alpha", "alpha1", "alpha2"],
        "beta": ["beta", "beta1", "beta2", "beta3"],
        "gamma": ["gamma", "gamma1", "gamma2"],
    }

    # Generate unique (high, low) band pairs from high over low, avoiding self- and reversed-duplicates
    selected_ratios = []
    main_bands = list(frequency_bands.keys())
    for i in range(len(main_bands)):
        for j in range(i):  # Ensure high over low only
            for high in frequency_bands[main_bands[i]]:
                for low in frequency_bands[main_bands[j]]:
                    selected_ratios.append((high, low))

    # Compute band ratios
    band_ratios = calculate_ratios(
        data_dict, fs, selected_ratios
    )  # Expected: {state: {(high, low): value}}

    # Convert to DataFrame for Plotly
    df = pd.DataFrame(band_ratios).T
    df.columns = [f"{high}/{low}" for high, low in df.columns]

    # Plot heatmap
    fig = px.imshow(
        df,
        labels=dict(x="Band Ratio", y="State", color="Value"),
        x=df.columns,
        y=df.index,
        color_continuous_scale="Viridis",
        aspect="auto",
    )

    fig.update_layout(
        title=f"{name}'s Band Power Ratios Heatmap - Channel {channel}"
        if name
        else "Band Power Ratios Heatmap",
        xaxis_tickangle=45,
        height=500,
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.show()
