import numpy as np
import librosa
import sqlite3
from pathlib import Path

# CREPE
import crepe

TARGET_SR = None  # keep librosa default behavior with sr=None to preserve original

def extract_audio_features(y, sr):
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # CREPE pitch
    y32 = y.astype(np.float32)

    try:
        time, frequency, confidence = crepe.predict(
            y32, sr, step_size=10, viterbi=True, model_capacity='small'
        )
    except:
        time = np.array([], dtype=np.float32)
        frequency = np.array([], dtype=np.float32)
        confidence = np.array([], dtype=np.float32)

    # median pitch
    if frequency.size and confidence.size:
        mask = confidence >= 0.2
        f_masked = frequency[mask]
        pitch_median = float(np.median(f_masked)) if f_masked.size else 0.0
    else:
        pitch_median = 0.0

    return tempo, mfcc.astype(np.float32), chroma.astype(np.float32), \
           time.astype(np.float32), frequency.astype(np.float32), confidence.astype(np.float32), pitch_median


def insert_audio_features(db_path, track_id, tempo, mfcc, chroma,
                          pitch_times=None, pitch_freqs=None, pitch_conf=None, pitch_median=0.0):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO audio_features (
            track_id, tempo, mfcc, chroma, pitch_times, pitch_freqs, pitch_conf, pitch_median
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (track_id,
          float(tempo),
          mfcc.tobytes(),
          chroma.tobytes(),
          (pitch_times.tobytes() if pitch_times is not None else None),
          (pitch_freqs.tobytes() if pitch_freqs is not None else None),
          (pitch_conf.tobytes() if pitch_conf is not None else None),
          float(pitch_median)
          ))

    conn.commit()
    conn.close()
