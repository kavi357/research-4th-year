# preprocess.py

import librosa
import numpy as np
import hashlib


TARGET_SR = 48000
TARGET_DURATION = 60.0  # seconds

def preprocess_audio(in_path):
    """
    Load audio, resample to 48kHz, convert to mono,
    trim silence, force 60s length, normalize to -20 dBFS
    """
    y, sr = librosa.load(str(in_path), sr=TARGET_SR, mono=True, res_type="kaiser_fast")

    # Trim silence
    y_trim, _ = librosa.effects.trim(y, top_db=25)

    # Force EXACT 60 seconds
    target_samples = int(TARGET_SR * TARGET_DURATION)
    if len(y_trim) > target_samples:
        y_out = y_trim[:target_samples]
    else:
        y_out = np.pad(y_trim, (0, target_samples - len(y_trim)), mode="constant")

    # Normalize to -20 dBFS
    rms = np.sqrt(np.mean(y_out**2) + 1e-12)
    y_out = y_out * (10**(-20 / 20) / rms)

    y_out = np.clip(y_out, -1.0, 1.0).astype(np.float32)

    return y_out, TARGET_DURATION, TARGET_SR




def compute_audio_hash(chroma, tempo, duration):
    """
    Robust perceptual identity hash.
    NOT used for similarity – only exact/near-exact identity.
    """

    # chroma shape: (12, T)
    chroma_mean = chroma.mean(axis=1)

    # Quantize for robustness
    chroma_q = np.round(chroma_mean, 2)
    tempo_q = round(float(tempo), 1)
    duration_q = round(float(duration), 1)

    payload = np.concatenate([
        chroma_q,
        np.array([tempo_q, duration_q], dtype=np.float32)
    ])

    return hashlib.sha1(payload.tobytes()).hexdigest()
