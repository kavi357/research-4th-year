
#backend/rhythm/rhythm_fingerprint.py
import numpy as np
import librosa

def extract_rhythm_fingerprint(y, sr, bars=8, beats_per_bar=4):
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
    beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")[1]

    needed = bars * beats_per_bar
    if len(beat_frames) < needed:
        return None

    beat_frames = beat_frames[:needed]
    fingerprint = []

    for i in range(len(beat_frames) - 1):
        s, e = beat_frames[i], beat_frames[i+1]
        fingerprint.append(1 if np.any((onset_frames >= s) & (onset_frames < e)) else 0)

    return np.array(fingerprint, dtype=np.int8)
