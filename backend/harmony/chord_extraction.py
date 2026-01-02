import numpy as np
import librosa

# 24 major/minor + N
CHORDS = [
    "C","Cm","C#","C#m","D","Dm","D#","D#m","E","Em",
    "F","Fm","F#","F#m","G","Gm","G#","G#m","A","Am","A#","A#m","B","Bm","N"
]

def chroma_to_chord(chroma):
    major = chroma
    minor = np.roll(chroma, 3)

    maj_idx = np.argmax(major)
    min_idx = np.argmax(minor)

    if major[maj_idx] >= minor[min_idx]:
        return maj_idx * 2     # even = major
    else:
        return min_idx * 2 + 1 # odd = minor


def extract_chord_sequence(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    beats = librosa.beat.beat_track(y=y, sr=sr)[1]

    chords = []
    beat_idx = []

    for i in beats:
        if i < chroma.shape[1]:
            chord = chroma_to_chord(chroma[:, i])
            if not chords or chord != chords[-1]:
                chords.append(chord)
                beat_idx.append(int(i))

    return np.array(chords, dtype=np.int8), np.array(beat_idx, dtype=np.int16)
