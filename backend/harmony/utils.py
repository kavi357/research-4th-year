# ================================
# Roman Numeral Conversion
# ================================


#backend/harmony/utils.py

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
ROMAN_MAP = {
    0: "I",
    2: "ii",
    4: "iii",
    5: "IV",
    7: "V",
    9: "vi",
    11: "vii°"
}

def chord_to_roman(chord_pc, key_pc):
    """
    chord_pc: pitch class of chord root (0-11)
    key_pc: detected tonic pitch class
    """
    interval = (chord_pc - key_pc) % 12

    if interval not in ROMAN_MAP:
        return "X"  # non-diatonic

    return ROMAN_MAP[interval]


def convert_to_roman_sequence(chord_sequence, key_pc):
    """
    chord_sequence: list[int]
    """
    return [chord_to_roman(c, key_pc) for c in chord_sequence]
