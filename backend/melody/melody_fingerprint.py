
#backend/melody/melody_fingerprint
import numpy as np

def extract_melody_fingerprint(midi_seq, max_len=64):
    """
    Returns (intervals, contour)
    """
    if midi_seq is None or len(midi_seq) < 5:
        return None, None

    midi_seq = midi_seq[:max_len]

    intervals = np.diff(midi_seq)
    contour = np.sign(intervals)  # -1, 0, +1

    # Clip large jumps (legal abstraction)
    intervals = np.clip(intervals, -12, 12)

    return intervals.astype(np.int8), contour.astype(np.int8)
