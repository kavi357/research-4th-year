import librosa
import numpy as np

def extract_siamese_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # MFCC (40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = mfcc.mean(axis=1).flatten()  # Ensure 1D

    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1).flatten()  # Ensure 1D

    # Spectral features (FORCE SCALARS)
    spec_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    spec_rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    # Combine all into single 1D vector
    base = np.concatenate([
        mfcc_mean,          # 40
        chroma_mean,        # 12
        np.array([
            spec_centroid,
            spec_bandwidth,
            spec_rolloff,
            zcr,
            tempo
        ], dtype=np.float32)
    ])

    # Pad / trim to EXACT 1570
    if base.shape[0] < 1570:
        base = np.pad(base, (0, 1570 - base.shape[0]))
    else:
        base = base[:1570]

    return base.astype(np.float32)
