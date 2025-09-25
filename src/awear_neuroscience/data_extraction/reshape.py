from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from awear_neuroscience.data_extraction.constants import (FIELD_KEYS,
                                                          SAMPLING_RATE)


def construct_long_df(
    raw_data: List[np.ndarray],
    timestamps: List[str],
    utc_timestamps: List[pd.Timestamp],
    focus_type_list: List[str],
    extra_metadata: List[Dict[str, Any]] = None,
    document_names: Optional[List[Optional[str]]] = None,
    session_ids: Optional[List[Optional[Any]]] = None,
    fs: int = SAMPLING_RATE,
) -> pd.DataFrame:
    """
    Construct a long-format DataFrame from processed EEG data.

    Args:
        raw_data: List of waveform arrays.
        timestamps: List of raw timestamp strings.
        utc_timestamps: List of UTC datetime objects.
        focus_type_list: List of focus type labels.
        document_names: Optional list of document_name values, one per segment.
        session_ids: Optional list of session_id values, one per segment.
        fs: Sampling frequency.

    Returns:
        Long-format DataFrame suitable for analysis or storage.
    """
    n_samples = SAMPLING_RATE
    n_segments = len(raw_data)

    long_df = pd.DataFrame(
        {
            "waveform_value": np.concatenate(raw_data),
            "segment": np.repeat([f"seg_{i}" for i in range(n_segments)], n_samples),
            "time_UTC": np.repeat(utc_timestamps, n_samples),
            "timestamp": np.repeat(timestamps, n_samples),
            "time_sample": np.tile(np.arange(n_samples) / n_samples, n_segments),
            "focus_type": np.repeat(focus_type_list, n_samples),
        }
    )

    # add optional document_name column
    if set(document_names) != {None}:
        long_df["document_name"] = np.repeat(document_names, fs)

    # add optional session_id column
    if set(session_ids) != {None}:
        long_df["session_id"] = np.repeat(session_ids, fs)

    if extra_metadata:
        extra_df = pd.DataFrame(extra_metadata)
        extra_df = extra_df.loc[
            :, [k for k in extra_df.columns if k in FIELD_KEYS and k != "timestamp"]
        ]
        extra_df["segment"] = [f"seg_{i}" for i in range(n_segments)]
        long_df = long_df.merge(extra_df, on="segment", how="left")

    return long_df


def normalize_session(session):
    def ensure_seconds(t_str):
        """Ensures time string has seconds."""
        return t_str + ":00" if len(t_str.split(":")) == 2 else t_str

    def parse_time(t_str):
        """Parses a time string, handling both 24h and 12h AM/PM formats."""
        t_str = t_str.strip()
        if "AM" in t_str.upper() or "PM" in t_str.upper():
            # Parse AM/PM format
            return datetime.strptime(t_str.upper(), "%I:%M %p").time()
        elif len(t_str.split(":")) == 2:
            # Add seconds if missing
            return datetime.strptime(t_str, "%H:%M").time()
        else:
            return datetime.strptime(t_str, "%H:%M:%S").time()

    def fmt(dt):
        """Formats datetime with microseconds."""
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    # Parse inputs
    ts = datetime.fromisoformat(session["timestamp"].replace("Z", "+00:00"))
    duration = session["duration_minutes"]
    date = ts.date()

    # Ensure seconds
    try:
        start_t = parse_time(session["start_time"])
        end_t = parse_time(session["end_time"])

        start_dt = datetime.combine(date, start_t)
        end_dt = datetime.combine(date, end_t)
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
    except Exception as e:
        raise ValueError(f"Invalid start or end time: {e}")

    # Calculate the actual duration between user-provided times
    empirical_duration = (end_dt - start_dt).total_seconds() / 60

    # Apply inference rules
    if (
        duration == 0.5
        or abs(empirical_duration - duration) > 0.01
        or end_dt > ts  # floating-point tolerance
    ):
        end_dt = ts
        start_dt = end_dt - timedelta(minutes=duration)

    time_ranges = [(start_dt, end_dt)]
    return start_dt, end_dt, fmt(start_dt), fmt(end_dt), time_ranges
