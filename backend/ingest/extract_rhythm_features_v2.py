##backend/ingest/extract_rhythm_features_v2

from rhythm.common_rhythm import extract_common_rhythm
from rhythm.rhythm_fingerprint import extract_rhythm_fingerprint

def extract_rhythm_v2(y, sr):
    common = extract_common_rhythm(y, sr)
    fp = extract_rhythm_fingerprint(y, sr)

    if fp is None:
        return None, None

    return common, fp
