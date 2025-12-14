import numpy as np
import librosa
import crepe
import sqlite3

TARGET_SR = 16000
MAX_SECONDS = 20
CONF_THRESHOLD = 0.2


def preprocess_for_crepe(y, sr):
    y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    max_samples = TARGET_SR * MAX_SECONDS
    if len(y) > max_samples:
        y = y[:max_samples]

    return y.astype(np.float32), TARGET_SR


def extract_crepe_pitch(y, sr):
    y32 = y.astype(np.float32)

    try:
        time, frequency, confidence, _ = crepe.predict(
            audio=y32,
            sr=sr,
            model_capacity='small',
            step_size=5,
            viterbi=False
        )
    except Exception:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            0.0
        )

    # FILTER LOW-CONFIDENCE FRAMES
    mask = confidence >= CONF_THRESHOLD

    time = time[mask]
    frequency = frequency[mask]
    confidence = confidence[mask]

    pitch_median = float(np.median(frequency)) if frequency.size else 0.0

    return (
        time.astype(np.float32),
        frequency.astype(np.float32),
        confidence.astype(np.float32),
        pitch_median
    )


def update_crepe_features(db_path, track_id, pitch_times, pitch_freqs, pitch_conf, pitch_median):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        UPDATE audio_features
        SET pitch_times = ?, pitch_freqs = ?, pitch_conf = ?, pitch_median = ?
        WHERE track_id = ?
    """, (
        pitch_times.tobytes(),
        pitch_freqs.tobytes(),
        pitch_conf.tobytes(),
        float(pitch_median),
        track_id
    ))

    conn.commit()
    conn.close()
