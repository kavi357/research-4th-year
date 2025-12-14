import numpy as np
import librosa
import crepe
from pathlib import Path

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
AUDIO_PATH = r"D:\SLIIT\gtzan songs\classical\classical.00000.wav"   # ← CHANGE THIS TO YOUR AUDIO FILE
MODEL_PATH = "models//crepe/crepe_small.keras"  # ← Your saved model
MAX_SECONDS = 20                         # CREPE-safe limit
TARGET_SR = 16000                        # CREPE requirement
CONF_THRESHOLD = 0.2                     # Minimum confidence
# -----------------------------------------------


def load_audio(path):
    print(f"\nLoading audio: {path}")
    y, sr = librosa.load(path, sr=None, mono=True)

    # Resample → 16 kHz
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # Trim if too long
    max_samples = TARGET_SR * MAX_SECONDS
    if len(y_16k) > max_samples:
        y_16k = y_16k[:max_samples]

    return y_16k.astype(np.float32), TARGET_SR


def test_crepe(y, sr):
    print("\nRunning CREPE pitch extraction...\n")

    print("Loading saved CREPE model:", MODEL_PATH)
    model = crepe.core.build_and_load_model(model_capacity='small')
    model.load_weights(MODEL_PATH)

    # Run CREPE
    result = crepe.predict(
        audio=y,
        sr=sr,
        model_capacity='small',
        step_size=10,
        viterbi=True
    )

    # Handle 3-output or 4-output models
    if len(result) == 3:
        time, frequency, confidence = result
    elif len(result) == 4:
        time, frequency, confidence, activation = result
    else:
        raise ValueError(f"Unexpected CREPE output length: {len(result)}")

    print("CREPE returned:")
    print("    time:", time.shape)
    print("    frequency:", frequency.shape)
    print("    confidence:", confidence.shape)

    # Filter by confidence
    mask = confidence >= CONF_THRESHOLD
    valid_freq = frequency[mask]

    pitch_median = float(np.median(valid_freq)) if valid_freq.size else 0.0

    print("\n===== RESULTS =====")
    print("Raw pitch count:", len(frequency))
    print("Valid pitch (conf >= 0.2):", len(valid_freq))
    print("Median pitch:", pitch_median)
    print("====================\n")

    # Print first few values
    print("Sample pitch values:")
    for i in range(min(10, len(frequency))):
        print(f"{i}: freq={frequency[i]:.2f}, conf={confidence[i]:.3f}")

    return time, frequency, confidence, pitch_median



if __name__ == "__main__":
    # Load audio
    y, sr = load_audio(AUDIO_PATH)

    # Run CREPE
    test_crepe(y, sr)
