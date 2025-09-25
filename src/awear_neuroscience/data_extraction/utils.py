from datetime import datetime, timezone
from typing import Union

import pandas as pd
from dateutil.parser import parse

# Controlling Wildcard Imports
__all__ = ["format_firestore_timestamp", "convert_string_to_utc_timestamp"]


def format_firestore_timestamp(dt: Union[datetime, pd.Timestamp]) -> str:
    """
    Format a datetime or pandas Timestamp into ISO8601 with microseconds and UTC zone,
    as required by Firestore queries.
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="microseconds")


def convert_string_to_utc_timestamp(ts_str: str) -> float:
    """Parse various ISO8601 timestamp strings and return UTC unix timestamp."""
    dt = parse(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.timestamp()


from datetime import datetime
from typing import List, Dict

def reformat_session_times(sessions: List[Dict]) -> List[Dict]:
    time_formats = [
        "%Y-%m-%d %I:%M %p",  # 12-hour format with AM/PM
        "%Y-%m-%d %H:%M:%S",  # 24-hour format with seconds
        "%Y-%m-%d %H:%M",     # 24-hour format without seconds
    ]

    def try_parse(datetime_str: str) -> str:
        for fmt in time_formats:
            try:
                return datetime.strptime(datetime_str, fmt).isoformat()
            except ValueError:
                continue
        raise ValueError(f"Time data '{datetime_str}' does not match known formats.")

    for session in sessions:
        timestamp = session.get('timestamp')
        date_str = timestamp.split('T')[0]  # Extract 'YYYY-MM-DD'

        session['start_time'] = try_parse(f"{date_str} {session['start_time']}")
        session['end_time'] = try_parse(f"{date_str} {session['end_time']}")

    return sessions


def reformat_session_times_df(df):
    def parse_time(row, time_col):
        date_str = row['timestamp'].split('T')[0]
        time_str = row[time_col]
        try:
            # Try parsing with AM/PM first
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %I:%M %p")
        except ValueError:
            try:
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                except ValueError:
                    dt = pd.NaT  # If all parsing fails
        return dt.isoformat() if pd.notna(dt) else None

    df['start_time'] = df.apply(lambda row: parse_time(row, 'start_time'), axis=1)
    df['end_time'] = df.apply(lambda row: parse_time(row, 'end_time'), axis=1)
    return df


# Assuming your DataFrame is called df
def merge_types(row):
    if pd.isna(row['focus_type']) and pd.isna(row['session_type']):
        return 'Focused'
    elif pd.notna(row['focus_type']) and pd.isna(row['session_type']):
        return row['focus_type']
    elif pd.isna(row['focus_type']) and pd.notna(row['session_type']):
        return row['session_type']
    else:
        # If both exist, choose one (e.g., prioritize focus_type)
        print(row)
        print('BOTH EXISTS')
        return row['focus_type']


def rename_focus_to_session(sessions):
    for session in sessions:
        if 'focus_type' in session and 'session_type' not in session:
            session['session_type'] = session.pop('focus_type')
    return sessions


from datetime import datetime

def parse_time(t_str: str) -> datetime:
    """
    Parse either a full ISO datetime or a time-only string.
    """
    try:
        # Try full ISO format first
        return datetime.fromisoformat(t_str)
    except ValueError:
        try:
            # Try time-only format
            return datetime.strptime(t_str, "%H:%M:%S").time()
        except ValueError:
            return datetime.strptime(t_str, "%H:%M").time()
