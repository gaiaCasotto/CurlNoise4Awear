"""
Data extraction constants for EEG Firestore pipeline.
"""
WAVEFORM_KEY = "waveformRIGHT_TEMP"
SAMPLING_RATE = 256

FIELD_KEYS = [
    "timestamp",
    "FOCUS_ist_dB",
    "GLOBAL_FOCUS_max_dB",
    "GLOBAL_FOCUS_min_dB",
    "GLOBAL_TABR_max_dB",
    "GLOBAL_TABR_min_dB",
    "GLOBAL_TGA_max_dB",
    "GLOBAL_TGA_min_dB",
    "TABR_avg_dB",
    "TABR_ist_dB",
    "TGA_avg_dB",
    "TGA_ist_dB",
]
