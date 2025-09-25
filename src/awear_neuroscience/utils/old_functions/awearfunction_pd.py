import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyentrp import entropy as ent
from scipy.integrate import simpson
from scipy.signal import butter, detrend, filtfilt, iirnotch, welch
# import difflib
from tensorpac import Pac

AWEAR_COLOR_SCHEME = [
    "rgb(97, 59, 209)",
    "rgb(215, 42, 19)",
    "rgb(72, 219, 202)",
    "rgb(245, 181, 25)",
    "rgb(180, 180, 180)",
]


def load_eeg_files(directory: str, columns: list) -> tuple[pd.DataFrame, int, dict]:
    """
    Loads all EEG .txt files in `directory`, reads only `columns`,
    extracts metadata from the first file, and returns:
      - a concatenated pandas DataFrame
      - the sampling rate (int)
      - the metadata dict
    """
    # 1) find all files
    paths = glob.glob(os.path.join(directory, "*.txt"))
    if not paths:
        return pd.DataFrame(), 0, {}

    # 2) extract metadata from first file
    metadata = {}
    with open(paths[0], "r", encoding="utf-8") as f:
        for _ in range(4):
            line = f.readline().strip()
            if "=" in line:
                k, v = line.split("=", 1)
                metadata[k.strip()] = v.strip()
            else:
                metadata[line] = None
    sample_rate = int(metadata.get("%Sample Rate", "0").split()[0])

    # 3) load each into a pandas DataFrame
    dfs = []
    for path in paths:
        df = pd.read_csv(path, skiprows=4, usecols=columns)
        fn = os.path.basename(path).split(".")[0].replace(" ", "")
        df = df.assign(filename=fn, time_s=np.arange(len(df)) / sample_rate)
        dfs.append(df)

    # 4) concat & clean up
    combined = pd.concat(dfs, ignore_index=True)
    # strip spaces in string columns (except filename) and cast to float where possible
    for col in combined.select_dtypes(include="object"):
        if col != "filename":
            combined[col] = (
                combined[col].str.replace(" ", "").astype(float, errors="ignore")
            )
    # strip leading spaces from column names
    combined.rename(columns=lambda x: x.lstrip(), inplace=True)

    return combined, sample_rate, metadata


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4
) -> np.ndarray:
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def notch_filter(data: np.ndarray, freq: float, fs: int, Q: float = 10) -> np.ndarray:
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data)


def apply_filters(
    df: pd.DataFrame,
    column: str,
    fs: int,
    bandpass_range: tuple[float, float] = (0.5, 47),
    notch_freqs: list[float] = [50, 60],
    order: int = 4,
    Q: float = 10,
) -> pd.DataFrame:
    """
    For each filename-group, filters `column` with bandpass + notch(s) + detrend,
    and appends a new column `<column>_filtered`.
    """

    def _filter(group: pd.DataFrame) -> pd.DataFrame:
        data = group[column].to_numpy()
        # bandpass
        filtered = butter_bandpass_filter(data, *bandpass_range, fs, order)
        # notch(s)
        for f0 in notch_freqs:
            filtered = notch_filter(filtered, f0, fs, Q)
        # detrend
        filtered = detrend(filtered)
        group = group.copy()
        group[f"{column}_filtered"] = filtered
        return group

    return df.groupby("filename", group_keys=False).apply(_filter)


def drop_first_n_seconds(
    df: pd.DataFrame, time_column: str = "time_s", n_seconds: int = 60
) -> pd.DataFrame:
    """
    Marks rows where `time_column` < n_seconds as to_drop=True.
    """
    return df.assign(to_drop=df[time_column] < n_seconds)


def remove_artifact(
    df: pd.DataFrame,
    column: str,
    fs: int,
    segment_length: int = 1,
    threshold: float = 60,
) -> pd.DataFrame:
    """
    - Creates integer segments by floor(time_s/segment_length).
    - Computes peak_signal per segment and flags artifact segments.
    - Computes corrected time ignoring dropped/artefacted samples.
    """
    # 1) segment numbering
    df = df.assign(
        segment_number=(
            df["filename"] + (df["time_s"] // segment_length).astype(int).astype(str)
        )
    )
    # 2) peak_signal per segment
    df["peak_signal"] = df.groupby("segment_number")[column].transform(
        lambda x: np.max(np.abs(x))
    )
    # 3) flag artifacts
    df["artifact"] = df["peak_signal"] >= threshold

    # 4) corrected time: only count samples that are not dropped/artifact
    valid = (~df["to_drop"]) & (~df["artifact"])
    df["time_corrected_s"] = valid.cumsum() / fs
    # shift so first valid sample is at t=−1/fs → + time for first sample at 0
    df["time_corrected_s"] -= 1.0 / fs

    return df.drop(columns=["segment_number"])


def compute_psd_file(
    df: pd.DataFrame,
    column: str,
    epoch_length: float,
    epoch_overlap: float,
    fs: int,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
) -> pd.DataFrame:
    """
    Runs Welch PSD on sliding windows of length epoch_length with epoch_overlap,
    returns exploded DataFrame of freq/psd pairs.
    """
    results = []
    t0, tmax = df["time_corrected_s"].min(), df["time_corrected_s"].max()
    current, epoch = t0, 1
    while current < tmax - epoch_length:
        seg = df[
            (df["time_corrected_s"] >= current)
            & (df["time_corrected_s"] < current + epoch_length)
        ]
        if not seg.empty:
            freqs, psd = welch(
                seg[column], fs, nperseg=nperseg, noverlap=noverlap, window=window
            )
            temp = pd.DataFrame(
                {
                    "start_time": current,
                    "end_time": current + epoch_length,
                    "epoch": epoch,
                    "frequency": [freqs.tolist()],
                    "psd": [psd.tolist()],
                }
            )
            results.append(temp)
            epoch += 1
        current += epoch_length - epoch_overlap

    if not results:
        return pd.DataFrame(
            columns=["start_time", "end_time", "epoch", "frequency", "psd"]
        )
    psd_df = pd.concat(results, ignore_index=True)
    # explode lists into long form
    return psd_df.explode(["frequency", "psd"]).reset_index(drop=True)


def compute_psd(
    df: pd.DataFrame,
    column: str,
    epoch_length: float,
    epoch_overlap: float,
    fs: int,
    nperseg: int = 256,
    noverlap: int = 128,
    window: str = "hann",
) -> pd.DataFrame:
    """
    Applies compute_psd_file to each filename (dropping to_drop/artifact),
    and concatenates into one big PSD table.
    """
    dfs = []
    for fn, group in df.groupby("filename"):
        grp = group[(~group["artifact"]) & (~group["to_drop"])]
        try:
            psd_df = compute_psd_file(
                grp, column, epoch_length, epoch_overlap, fs, nperseg, noverlap, window
            ).assign(filename=fn)
            dfs.append(psd_df)
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            continue

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def apply_full_pipeline(
    directory_path: str,
    bandpass_range: list[float] = [1, 50],
    notch_freqs: list[float] = [50, 60],
    drop_initial_n_seconds: int = 60,
    epoch_length: int = 15,
    epoch_overlap: int = 13,
    threshold_artifacts: float = 60,
    columns: list[str] = [
        "Sample Index",
        " EXG Channel 0",
        " EXG Channel 1",
        " EXG Channel 2",
        " EXG Channel 3",
        " Timestamp",
    ],
    channel: str = "EXG Channel 0",
) -> tuple[pd.DataFrame, pd.DataFrame, int, dict]:
    """
    Runs the full EEG pipeline:
      1) load files
      2) bandpass + notch + detrend
      3) drop initial seconds
      4) segment & remove artifacts
      5) compute PSD
    Returns:
      df_final (with time_corrected_s, artifact flags, etc.),
      psd (PSD table),
      fs (sampling rate),
      metadata (dict)
    """
    # assume these helper functions were rewritten to use pandas:
    df, fs, metadata = load_eeg_files(directory_path, columns)
    df = apply_filters(df, channel, fs, bandpass_range, notch_freqs)
    df = drop_first_n_seconds(df, "time_s", drop_initial_n_seconds)
    df_final = remove_artifact(
        df, f"{channel}_filtered", fs, segment_length=1, threshold=threshold_artifacts
    )
    psd = compute_psd(
        df_final,
        column=f"{channel}_filtered",
        epoch_length=epoch_length,
        epoch_overlap=epoch_overlap,
        fs=fs,
        nperseg=fs,
        noverlap=fs // 2,
        window="hann",
    )
    return df_final, psd, fs, metadata


def plot_ts_and_psd(
    df_final: pd.DataFrame,
    psd: pd.DataFrame,
    filename: str,
    channel: str = "EXG Channel 0",
    yaxes_range: list[float] | None = None,
    width: int = 1200,
    height: int = 600,
    showlegend: bool = False,
) -> go.Figure:
    """
    For one recording (filename), plots:
      • left: time-series filtered signal colored by artifact retention
          (red = dropped, green = kept)
      • right: PSD curves (log-y), one line per epoch, yellow→blue scale,
          legend shown only for epochs 0, 5, 10, …
    """
    # time-series prep
    ts_fig = df_final[df_final["filename"] == filename].copy()
    ts_fig = ts_fig.sort_values("artifact")
    ts_fig["to_keep"] = (~ts_fig["artifact"]) & (~ts_fig["to_drop"])

    # PSD prep
    psd_fig = psd[psd["filename"] == filename].copy()
    epochs = sorted(psd_fig["epoch"].unique())
    n_epochs = len(epochs)
    frac = [i / (n_epochs - 1) if n_epochs > 1 else 0.5 for i in range(n_epochs)]
    colors = px.colors.sample_colorscale("YlGnBu", frac)
    epoch_color_map = {str(e): c for e, c in zip(epochs, colors)}
    psd_fig["epoch_str"] = psd_fig["epoch"].astype(str)

    # create subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[f"{filename} - {channel}", "PSD"]
    )

    # left: time-series scatter
    ts = px.scatter(
        ts_fig,
        x="time_s",
        y=f"{channel}_filtered",
        color="to_keep",
        opacity=0.7,
        color_discrete_map={False: "red", True: "green"},
        labels={"to_keep": "Kept segment"},
    )
    for tr in ts.data:
        fig.add_trace(tr, row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Filtered Signal (µV)", row=1, col=1)

    # right: PSD lines with selective legend
    ps = px.line(
        psd_fig,
        x="frequency",
        y="psd",
        color="epoch_str",
        color_discrete_map=epoch_color_map,
        labels={"epoch_str": "Epoch"},
    )
    for tr in ps.data:
        # show legend only for epochs 0,5,10,...
        epoch = int(tr.name)
        tr.showlegend = epoch % 5 == 0
        fig.add_trace(tr, row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 47], row=1, col=2)
    fig.update_yaxes(
        title_text="PSD (µV²/Hz)", type="log", range=yaxes_range, row=1, col=2
    )

    # layout
    fig.update_layout(showlegend=showlegend, width=width, height=height)
    return fig


def selfreport_analysis(
    df: pd.DataFrame,
    columns: list[str] = [
        "Self-report (Likert 1-9)_Valence",
        "Self-report (Likert 1-9)_Arousal",
    ],
    name_col: str = "filename",
    candidate: str = "s",
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Finds—and labels—the four corner cases (LVLA, LVHA, HVLA, HVHA) plus the centroid
    from two self-report axes, and plots them on a scatter.
    Returns a DataFrame of the selected cases with their IDs and quadrant labels.
    """
    # match requested columns to actual df columns
    actual = df.columns
    cmap = {
        col: (difflib.get_close_matches(col, actual, n=1, cutoff=0.0) or [None])[0]
        for col in columns
    }
    xcol, ycol = cmap[columns[0]], cmap[columns[1]]

    # compute bounds & centroid
    xmin, xmax = df[xcol].min(), df[xcol].max()
    ymin, ymax = df[ycol].min(), df[ycol].max()
    xavg, yavg = df[xcol].mean(), df[ycol].mean()

    corners = {
        "low_valence_low_arousal": (xmin, ymin),
        "low_valence_high_arousal": (xmin, ymax),
        "high_valence_low_arousal": (xmax, ymin),
        "high_valence_high_arousal": (xmax, ymax),
    }

    # find index of nearest point to each corner & centroid
    results = {}
    for lbl, (cx, cy) in {**corners, "neutral": (xavg, yavg)}.items():
        dist = np.hypot(df[xcol] - cx, df[ycol] - cy)
        results[lbl] = dist.idxmin()

    # label mapping
    pm = {
        "low_valence_low_arousal": ("low", "low", "LVLA", "SAD"),
        "low_valence_high_arousal": ("low", "high", "LVHA", "TENSE"),
        "high_valence_low_arousal": ("high", "low", "HVLA", "CALM"),
        "high_valence_high_arousal": ("high", "high", "HVHA", "HAPPY"),
        "neutral": ("neutral", "neutral", "Neutral", "Neutral"),
    }

    rows = []
    for lbl, idx in results.items():
        uid = df.at[idx, name_col]
        v, a, quad, emo = pm[lbl]
        rows.append(
            {
                "filename": uid,
                "valence": v,
                "arousal": a,
                "quadrant": quad,
                "emotion": emo,
            }
        )
    out = pd.DataFrame(rows).merge(
        df[[name_col, xcol, ycol]].rename(
            columns={xcol: "val_value", ycol: "aro_value"}
        ),
        on="filename",
    )

    # scatter plot
    fig = px.scatter(
        df,
        x=xcol,
        y=ycol,
        opacity=0.5,
        hover_name=name_col,
        color="Categories",
        title=f"{candidate} ‑ Self reports",
    )
    fig.update_traces(marker=dict(size=10))
    sel = df[name_col].isin(out["filename"])
    fig.add_trace(
        go.Scatter(
            x=df.loc[sel, xcol],
            y=df.loc[sel, ycol],
            mode="markers+text",
            marker=dict(color="black", size=10, symbol="x"),
            name="selected",
        )
    )
    fig.update_xaxes(title_text="Valence")
    fig.update_yaxes(title_text="Arousal")
    fig.update_layout(showlegend=True, width=700, height=500)

    return out, fig


def compare_psd_corners(
    psd: pd.DataFrame,
    corner_cases: pd.DataFrame,
    filter_values: list[str],
    plot_in_db: bool = True,
    awear_color_scheme: bool = True,
    last_N_epochs: int = 1000,
) -> tuple[go.Figure, go.Figure]:
    """
    For each quadrant in `filter_values`, pulls its filename from `corner_cases`,
    then:
      • fig1: overlays each epoch’s PSD line
      • fig2: plots mean±std band per quadrant
    Returns (fig1, fig2).
    """
    fig1 = go.Figure()
    fig2 = go.Figure()
    palette = px.colors.qualitative.Plotly
    AWEAR = palette  # replace with your AWEAR_COLOR_SCHEME if defined
    psd["epoch"] = psd["epoch"].astype(int)
    psd_filtered = psd.groupby(["filename", "frequency"], group_keys=False).apply(
        lambda g: g.nlargest(last_N_epochs, "epoch")
    )

    for i, quad in enumerate(filter_values):
        uid = corner_cases.loc[corner_cases["quadrant"] == quad, "filename"].iat[0]
        label = corner_cases.loc[corner_cases["filename"] == uid, "emotion"].iat[0]
        df_q = psd_filtered[psd_filtered["filename"] == uid].copy()
        col = "psd_dB" if plot_in_db else "psd"
        if plot_in_db:
            psd_vals = pd.to_numeric(df_q["psd"], errors="raise").to_numpy(dtype=float)
            df_q["psd_dB"] = 10 * np.log10(psd_vals + 1e-15)

        # stats per frequency
        stats = (
            df_q.groupby("frequency")[col]
            .agg(mean="mean", std="std", pt_count="count")
            .reset_index()
        )

        if awear_color_scheme:
            color = AWEAR_COLOR_SCHEME[i]
        else:
            color = px.colors.qualitative.Plotly[i]

        # all epochs
        for j, ep in enumerate(sorted(df_q["epoch"].unique())):
            sub = df_q[df_q["epoch"] == ep]
            fig1.add_trace(
                go.Scatter(
                    x=sub["frequency"],
                    y=sub[col],
                    mode="lines",
                    line=dict(width=2, color=color),
                    opacity=0.7,
                    name=label if j == 0 else None,
                    legendgroup=label,
                    showlegend=(j == 0),
                )
            )

        # mean line
        fig2.add_trace(
            go.Scatter(
                x=stats["frequency"],
                y=stats["mean"],
                mode="lines",
                line=dict(width=4, color=color),
                name=f"{label}_mean",
                legendgroup=label,
            )
        )

        # ±std band
        ub = stats["mean"] + stats["std"]
        lb = stats["mean"] - stats["std"]
        xs = np.concatenate([stats["frequency"], stats["frequency"][::-1]])
        ys = np.concatenate([ub, lb[::-1]])
        fig2.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                fillcolor=color,
                opacity=0.3,
                line=dict(color="rgba(255,255,255,0)"),  # Invisible line border
                hoverinfo="skip",
                name=f"{label}_±std",
                showlegend=True,
            )
        )

    # axis & layout
    for fig in (fig1, fig2):
        fig.update_xaxes(title="Frequency (Hz)", range=[0, 47])
    if plot_in_db:
        ytitle, yrange = "PSD (dB/Hz)", [-15, 3]
        fig1.update_yaxes(title=ytitle, range=yrange)
        fig2.update_yaxes(title=ytitle, range=yrange)
    else:
        ytitle, yrange = "PSD (µV²/Hz)", [-3, 2]
        fig1.update_yaxes(title=ytitle, range=yrange, type="log")
        fig2.update_yaxes(title=ytitle, range=yrange, type="log")

    fig1.update_layout(width=900, height=600)
    fig2.update_layout(width=900, height=600)

    return fig1, fig2


def extract_features_psd(
    psd: pd.DataFrame, frequency_ranges: dict[str, tuple[float, float]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From a PSD table with columns ['filename', 'epoch', 'frequency', 'psd'],
    compute, for each (filename, epoch):
      • normalized band powers per frequency_ranges
      • those band powers in dB
      • relative band powers
      • inter‐band ratios
    Returns:
      features:      DataFrame indexed by filename, epoch with all features
      avg_per_file:  mean of those features per filename
    """

    def integrate_spectrum(
        x: np.ndarray, y: np.ndarray, lower: float, upper: float
    ) -> float:
        mask = (x >= lower) & (x <= upper)
        x_f, y_f = x[mask], y[mask]
        if x_f.size < 2:
            return np.nan
        # ensure sorted
        order = np.argsort(x_f)
        return simpson(y_f[order], x_f[order])

    def integrate_and_norm(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
        val = integrate_spectrum(x, y, lo, hi)
        span = hi - lo
        return val / span if span > 0 else np.nan

    def features_per_group(group: pd.DataFrame) -> pd.Series:
        x = group["frequency"].to_numpy()
        y = group["psd"].to_numpy()

        # 1) normalized band powers
        norm_powers = {
            key: integrate_and_norm(x, y, lo, hi)
            for key, (lo, hi) in frequency_ranges.items()
        }

        # 2) same in dB
        db_powers = {
            f"{key}_dB": 10 * np.log10(1e-9 + norm_powers[key]) for key in norm_powers
        }

        # 3) total power over full spectrum
        tot = integrate_spectrum(x, y, x.min(), x.max())

        # 4) relative powers
        rel_powers = {f"{key}_relative": norm_powers[key] / tot for key in norm_powers}

        # combine
        feats = {**norm_powers, **db_powers, **rel_powers}
        s = pd.Series(feats)

        # 5) inter‑band ratios
        s["beta_to_delta"] = s["beta"] / s["delta"]
        s["beta_to_theta"] = s["beta"] / s["theta"]
        s["beta_to_alpha"] = s["beta"] / s["alpha"]
        s["gamma_to_delta"] = s["gamma"] / s["delta"]
        s["gamma_to_theta"] = s["gamma"] / s["theta"]
        s["gamma_to_alpha"] = s["gamma"] / s["alpha"]

        return s

    # apply per (filename, epoch)
    features = (
        psd.groupby(["filename", "epoch"], as_index=False)
        .apply(features_per_group)
        .reset_index(drop=True)
    )

    # average across epochs per filename
    avg_per_file = features.groupby("filename", as_index=False).mean()

    return features, avg_per_file


# def compute_pac_epoch(data, fs, frequency_ranges, frequency_pac, ipac=(2,0,0)):
#     pac_d={}
#     for high_fq_band, low_fq_band in frequency_pac:
#         low_fq_range = frequency_ranges[low_fq_band]
#         high_fq_range = frequency_ranges[high_fq_band]
#         p = Pac(idpac=ipac,        # Tort MI
#                 f_pha=low_fq_range,   # phase band
#                 f_amp=high_fq_range,  # amplitude band
#                 dcomplex='hilbert',

#         )
#         pac_d[f'pac_{low_fq_band}_{high_fq_band}'] = p.filterfit(sf=fs, x_pha=data, x_amp=data, verbose=False)[0][0][0]
#     return pd.Series(pac_d)

# def compute_pac_file(
#     df: pd.DataFrame,
#     column: str,
#     epoch_length: float,
#     epoch_overlap: float,
#     fs: int,
#     frequency_ranges: dict[str, tuple[float, float]],
#     frequency_pac: list[tuple[str, str]],
#     ipac: tuple[int, int, int] = (2, 0, 0)
# ) -> pd.DataFrame:
#     """
#     Runs evaluate_pac_epoch() on successive overlapping epochs of `column` in `df`.
#     Returns a DataFrame with:
#       - one row per epoch
#       - one column per (high_fq, low_fq) MI value
#       - columns 'epoch', 'start_time', 'end_time'
#     """
#     results = []
#     t0, tmax = df['time_corrected_s'].min(), df['time_corrected_s'].max()
#     current = t0
#     epoch = 1

#     while current < tmax - epoch_length:
#         # grab that epoch's samples
#         seg = df.loc[
#             (df['time_corrected_s'] >= current) &
#             (df['time_corrected_s'] < current + epoch_length),
#             column
#         ].to_numpy()

#         if seg.size > 0:
#             # compute PAC for this segment
#             pac_series = compute_pac_epoch(
#                 data=seg,
#                 fs=fs,
#                 frequency_ranges=frequency_ranges,
#                 frequency_pac=frequency_pac,
#                 ipac=ipac
#             )

#             # annotate with epoch metadata
#             pac_series['epoch'] = epoch
#             pac_series['start_time'] = current
#             pac_series['end_time']   = current + epoch_length

#             results.append(pac_series)
#             epoch += 1

#         # advance by (length − overlap)
#         current += (epoch_length - epoch_overlap)

#     # combine all into one DataFrame
#     if results:
#         return pd.DataFrame(results).reset_index(drop=True)
#     else:
#         # no epochs? return empty with columns
#         sample = compute_pac_epoch(
#             data=np.zeros(1),
#             fs=fs,
#             frequency_ranges=frequency_ranges,
#             frequency_pac=frequency_pac,
#             ipac=ipac
#         )
#         cols = list(sample.index) + ['epoch','start_time','end_time']
#         return pd.DataFrame(columns=cols)


# def compute_pac(
#     df: pd.DataFrame,
#     column: str,
#     epoch_length: float,
#     epoch_overlap: float,
#     fs: int,
#     frequency_ranges: dict,
#     frequency_pac: list[tuple[str, str]],
#     ipac: tuple[int, int, int] = (2, 0, 0)
# ) -> pd.DataFrame:
#     """
#     Applies compute_pac_file to each filename (dropping to_drop/artifact),
#     and concatenates into one big PSD table.
#     """
#     dfs = []
#     for fn, group in df.groupby('filename'):
#         grp = group[(~group['artifact']) & (~group['to_drop']) & (group['time_corrected_s']>=0)]
#         try:
#             psd_df = compute_pac_file(
#                 grp, column, epoch_length, epoch_overlap,
#                 fs, frequency_ranges, frequency_pac, ipac
#             ).assign(filename=fn)
#             dfs.append(psd_df)
#         except Exception as e:
#             print(f"Error processing {fn}: {e}")
#             continue
#     features=pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
#     # average across epochs per filename
#     avg_per_file = (
#         features
#         .groupby('filename', as_index=False)
#         .mean()
#     )

#     return features, avg_per_file


def compute_pac_epoch(data, fs, frequency_ranges, frequency_pac, ipac=(2, 0, 0)):
    pac_d = {}
    for low_fq_band, high_fq_band in frequency_pac:
        low_fq_range = frequency_ranges[low_fq_band]
        high_fq_range = frequency_ranges[high_fq_band]
        p = Pac(
            idpac=ipac,  # Tort MI
            f_pha=low_fq_range,  # phase band
            f_amp=high_fq_range,  # amplitude band
            dcomplex="hilbert",
        )
        pac_d[f"pac_{low_fq_band}_{high_fq_band}"] = p.filterfit(
            sf=fs, x_pha=data, x_amp=data, verbose=False
        )[0][0][0]
    return pd.Series(pac_d)


def compute_entropy(data, m=2, r_factor=0.2, max_scale=25):
    """
    Compute Shannon and Multiscale Entropy for a given 1D signal.

    Parameters:
    - data: 1D numpy array or list
    - m: embedding dimension for SampEn
    - r_factor: tolerance multiplier to signal std
    - max_scale: compute scales 1 through max_scale

    Returns:
    - H_shannon: Shannon entropy
    - mse_values: Multiscale entropy values at each scale
    """

    # Compute Shannon entropy
    H_shannon = ent.shannon_entropy(data)

    # Compute Multiscale Entropy
    r = r_factor * np.std(data)
    mse_values = ent.multiscale_entropy(data, m, r, max_scale)
    entropy_dict = {
        f"scale_{i+1}": mse
        for i, mse in enumerate(mse_values)
        if (i == 0) or ((i + 1) % 5 == 0)
    }
    entropy_dict["H_shannon"] = H_shannon
    return pd.Series(entropy_dict)


def extract_features_ts_file(
    df: pd.DataFrame,
    column: str,
    epoch_length: float,
    epoch_overlap: float,
    fs: int,
    frequency_ranges: dict[str, tuple[float, float]],
    frequency_pac: list[tuple[str, str]],
    ipac: tuple[int, int, int] = (2, 0, 0),
) -> pd.DataFrame:
    """
    Runs evaluate_pac_epoch() on successive overlapping epochs of `column` in `df`.
    Returns a DataFrame with:
      - one row per epoch
      - one column per (high_fq, low_fq) MI value
      - columns 'epoch', 'start_time', 'end_time'
    """
    results = []
    t0, tmax = df["time_corrected_s"].min(), df["time_corrected_s"].max()
    current = t0
    epoch = 1

    while current < tmax - epoch_length:
        # grab that epoch's samples
        seg = df.loc[
            (df["time_corrected_s"] >= current)
            & (df["time_corrected_s"] < current + epoch_length),
            column,
        ].to_numpy()

        if seg.size > 0:
            # compute PAC for this segment
            pac_series = compute_pac_epoch(
                data=seg,
                fs=fs,
                frequency_ranges=frequency_ranges,
                frequency_pac=frequency_pac,
                ipac=ipac,
            )

            # annotate with epoch metadata
            pac_series["epoch"] = epoch
            pac_series["start_time"] = current
            pac_series["end_time"] = current + epoch_length

            entropy = compute_entropy(seg, m=2, r_factor=0.2, max_scale=25)

            results.append(pd.concat([pac_series, entropy]))
            epoch += 1

        # advance by (length − overlap)
        current += epoch_length - epoch_overlap

    # combine all into one DataFrame
    if results:
        return pd.DataFrame(results).reset_index(drop=True)
    else:
        # no epochs? return empty with columns
        sample = compute_pac_epoch(
            data=np.zeros(1),
            fs=fs,
            frequency_ranges=frequency_ranges,
            frequency_pac=frequency_pac,
            ipac=ipac,
        )
        cols = list(sample.index) + ["epoch", "start_time", "end_time"]
        return pd.DataFrame(columns=cols)


def extract_features_ts(
    df: pd.DataFrame,
    column: str,
    epoch_length: float,
    epoch_overlap: float,
    fs: int,
    frequency_ranges: dict,
    frequency_pac: list[tuple[str, str]],
    ipac: tuple[int, int, int] = (2, 0, 0),
) -> pd.DataFrame:
    """
    Extract PAC and entropy features from a DataFrame of time series data.
    """
    feat_dfs = []
    for fn, group in df.groupby("filename"):
        grp = group[
            (~group["artifact"])
            & (~group["to_drop"])
            & (group["time_corrected_s"] >= 0)
        ]
        try:
            feat_df = extract_features_ts_file(
                grp,
                column,
                epoch_length,
                epoch_overlap,
                fs,
                frequency_ranges,
                frequency_pac,
                ipac,
            ).assign(filename=fn)
            feat_dfs.append(feat_df)
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            continue
    features = pd.concat(feat_dfs, ignore_index=True) if feat_dfs else pd.DataFrame()
    # average across epochs per filename
    avg_per_file = features.groupby("filename", as_index=False).mean()

    return features, avg_per_file
