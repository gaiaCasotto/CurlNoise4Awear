from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from google.cloud import firestore

from awear_neuroscience.data_extraction.constants import (FIELD_KEYS,
                                                          SAMPLING_RATE,
                                                          WAVEFORM_KEY)
from awear_neuroscience.data_extraction.reshape import (construct_long_df,
                                                        normalize_session)
from awear_neuroscience.data_extraction.utils import (
    convert_string_to_utc_timestamp, format_firestore_timestamp)

MAX_DURATION_MINUTES = 300


def query_eeg_data(
    firestore_client: firestore.Client,
    collection_name: str,
    document_name: str,
    subcollection_name: str,
    time_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
    chunk_size: timedelta = timedelta(minutes=15),
) -> List[Dict[str, Any]]:
    """
    Query EEG data from Firestore. Uses explicit time_ranges or auto-chunks.
    Returns a flat list of raw record dicts.
    """
    col_ref = firestore_client.collection(collection_name)
    subcol = col_ref.document(document_name).collection(subcollection_name)

    if time_ranges is None:
        now = datetime.utcnow()
        time_ranges = []
        start = now - timedelta(days=1)
        while start < now:
            end = min(start + chunk_size, now)
            time_ranges.append((start, end))
            start = end

    results: List[Dict[str, Any]] = []
    for start, end in time_ranges:
        start_ts = format_firestore_timestamp(start)
        end_ts = format_firestore_timestamp(end)
        query = (
            subcol.where("timestamp", ">=", start_ts)
            .where("timestamp", "<=", end_ts)
            .order_by("timestamp")
        )
        for doc in query.stream():
            results.append(doc.to_dict())
    return results


def get_selreport_data(
    firestore_client: firestore.Client,
    collection_name: str,
    document_name: str,
    time_ranges: list[tuple[datetime, datetime]],
    sessions_of_interest: List[str],
) -> List[Dict[str, Any]]:
    """
    Retrieve EEG data for selected focus sessions, normalized and annotated.

    Parameters
    ----------
    firestore_client : firestore.Client
        Initialized Firestore client.
    collection_name : str
        Name of the top‐level collection.
    document_name : str
        Document ID within the collection.
    time_ranges : Tuple[datetime, datetime]
        Time‐range filters to apply when querying sessions.
    sessions_of_interest : List[str]
        Session‐type names to include (case‐insensitive).

    Returns
    -------
    List[Dict[str, Any]]
        A flat list of EEG data points, each annotated with session_id and session_type.
    """

    sessions_of_interest = [s.lower() for s in sessions_of_interest]
    results: List[Dict[str, Any]] = []

    # 1) Fetch session metadata
    sessions_metadata = query_eeg_data(
        firestore_client=firestore_client,
        collection_name=collection_name,
        document_name=document_name,
        subcollection_name="focus_sessions",
        time_ranges=time_ranges,
    )
    if not sessions_metadata:
        return results  # nothing to do

    # 2) Loop through sessions, filter & load EEG
    session_id = 0
    for meta in sessions_metadata:
        duration = meta.get("duration_minutes", 0)
        session_type = meta.get("session_type", "").lower() if "session_type" in meta else meta.get("focus_type", "").lower()

        # Skip if too long or not in our interest list
        if duration >= MAX_DURATION_MINUTES or session_type not in sessions_of_interest:
            continue

        # Normalize to get the exact time ranges for this session
        *_, session_time_ranges = normalize_session(meta)

        # Fetch the live EEG data
        try:
            eeg_records = query_eeg_data(
                firestore_client=firestore_client,
                collection_name=collection_name,
                document_name=document_name,
                subcollection_name="live_data",
                time_ranges=session_time_ranges,
            )
        except Exception:
            print(
                f"Error querying live_data for session_id={session_id}, type={session_type}"
            )
            continue

        # Annotate and collect
        for record in eeg_records:
            record["session_id"] = session_id
            record["session_type"] = meta.get("session_type", "").lower() if meta["session_type"] else meta.get("focus_type", "").lower()
            record["document_name"] = document_name
            record["session_start"] = meta.get("start_time")
            record["session_end"] = meta.get("end_time")
            record["session_duration"] = meta.get("duration_minutes")
        results.extend(eeg_records)
        session_id += 1

    return results


def process_eeg_records(
    records: List[Dict[str, Any]], return_long: bool = False
) -> pd.DataFrame:
    """
    Transform raw Firestore records into structured or long-form DataFrame.

    Args:
        records: List of Firestore EEG record dictionaries.
        return_long: Whether to return long-format DataFrame for time-series analysis.

    Returns:
        pd.DataFrame: either a wide-format or long-format DataFrame.
    """
    raw_data = []
    timestamps = []
    utc_timestamps = []
    focus_type_list = []
    document_names = []
    session_ids = []
    extra_metadata = []

    for rec in records:
        wf = rec.get(WAVEFORM_KEY)
        if not isinstance(wf, list) or len(wf) != SAMPLING_RATE:
            continue
        raw_data.append(np.array(wf, dtype=np.float32))
        timestamps.append(rec["timestamp"])
        utc_timestamps.append(
            pd.to_datetime(
                convert_string_to_utc_timestamp(rec["timestamp"]), unit="s", utc=True
            )
        )
        focus_type_list.append(
            rec.get("focus_type") or rec.get("session_type", "no_label")
        )
        document_names.append(rec.get("document_name"))
        session_ids.append(rec.get("session_id"))
        extra_metadata.append(
            {k: v for k, v in rec.items() if k in FIELD_KEYS and k != "timestamp"}
        )

    if return_long:
        return construct_long_df(
            raw_data,
            timestamps,
            utc_timestamps,
            focus_type_list,
            extra_metadata,
            document_names,
            session_ids,
        )

    # Else return a simple wide-format structure
    rows = []
    for i in range(len(raw_data)):
        row = {
            "waveform": raw_data[i],
            "timestamp": timestamps[i],
            "utc_ts": utc_timestamps[i],
            "focus_type": focus_type_list[i],
        }
        # only include these keys if not None
        if document_names[i] is not None:
            row["document_name"] = document_names[i]
        if session_ids[i] is not None:
            row["session_id"] = session_ids[i]
        rows.append(row)
    return pd.DataFrame(rows)
