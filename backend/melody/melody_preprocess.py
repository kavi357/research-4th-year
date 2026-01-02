#backend/melody/melody_preprocess

import numpy as np
import librosa

def preprocess_melody(pitch_freqs, pitch_conf, conf_th=0.15):
    """
    Returns cleaned MIDI melody sequence
    """
    if pitch_freqs is None or len(pitch_freqs) == 0:
        return None

    mask = pitch_conf >= conf_th
    freqs = pitch_freqs[mask]

    if freqs.size < 5:
        return None

    midi = librosa.hz_to_midi(freqs)
    midi = np.round(midi).astype(int)

    return midi
