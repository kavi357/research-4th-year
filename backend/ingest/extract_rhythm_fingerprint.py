import numpy as np
import librosa

def extract_rhythm_fingerprint(y, sr, beats_per_bar=4, bars=8):
    """
    Copyright-oriented rhythm fingerprint.
    - Beat-normalized
    - Binary onset pattern
    - Captures rhythmic motifs
    """

    # 1. Onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # 2. Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr
    )

    if len(beat_frames) < beats_per_bar * bars:
        return None  # Not enough rhythmic content

    # 3. Convert beats to time
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # 4. Detect onsets
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # 5. Beat grid (quantized)
    total_beats = beats_per_bar * bars
    beat_grid = beat_times[:total_beats]

    # 6. Binary fingerprint
    fingerprint = np.zeros(total_beats, dtype=np.int8)

    for onset in onset_times:
        idx = np.argmin(np.abs(beat_grid - onset))
        if abs(beat_grid[idx] - onset) < 0.08:  # 80ms tolerance
            fingerprint[idx] = 1

    return fingerprint
