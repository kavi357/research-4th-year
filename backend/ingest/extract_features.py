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

    # ===== FIXED CREPE =====
    y_crepe = librosa.resample(y, orig_sr=sr, target_sr=16000)
    max_samples = 16000 * 20
    if len(y_crepe) > max_samples:
        y_crepe = y_crepe[:max_samples]

    y_crepe = y_crepe.astype(np.float32)

    try:
        time, frequency, confidence, _ = crepe.predict(
            audio=y_crepe,
            sr=16000,
            model_capacity='small',
            step_size=5,
            viterbi=False
        )
    except Exception:
        time = np.array([], dtype=np.float32)
        frequency = np.array([], dtype=np.float32)
        confidence = np.array([], dtype=np.float32)

    if confidence.size:
        mask = confidence >= 0.2
        time = time[mask]
        frequency = frequency[mask]
        confidence = confidence[mask]

    pitch_median = float(np.median(frequency)) if frequency.size else 0.0

    return (
        tempo,
        mfcc.astype(np.float32),
        chroma.astype(np.float32),
        time.astype(np.float32),
        frequency.astype(np.float32),
        confidence.astype(np.float32),
        pitch_median
    )



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
