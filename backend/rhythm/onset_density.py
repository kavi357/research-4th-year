#backend/rhythm/onset_density
import librosa
import numpy as np

def compute_onset_density(
    audio_path: str,
    sr: int = 22050,
    window_sec: float = 1.0,
    hop_sec: float = 0.5
) -> dict:
    """
    Compute rhythm onset density timeline
    """

    y, sr = librosa.load(audio_path, sr=sr)

    # Onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Convert time → frames
    window_frames = int(window_sec * sr / 512)
    hop_frames = int(hop_sec * sr / 512)

    densities = []
    times = []

    for i in range(0, len(onset_env) - window_frames, hop_frames):
        window = onset_env[i:i + window_frames]
        density = float(np.mean(window))

        densities.append(round(density, 5))
        times.append(round(librosa.frames_to_time(i, sr=sr), 2))

    return {
        "times": times,
        "densities": densities,
        "window_sec": window_sec,
        "hop_sec": hop_sec
    }
