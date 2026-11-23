"""Move classes.

Labels come from the segment files (data/raw/<video>.segments.csv), so a new dance
only needs new label names there and matching entries in presets/effects.json.
The list below is the "L을 가져가" trend dance we trained on; train.py puts "idle"
first and the rest in alphabetical order, and saves the order to <model>_labels.json.
"""

LABELS = [
    ("idle", "기타 (동작 아님, 인트로)"),
    ("L_gajyeoga", "L를 가져가"),
    ("cream_cheese_honey", "크림 치즈 허니"),
    ("criss_cross", "크리스 크로스"),
    ("selca", "셀카"),
    ("yachae", "야채"),
]

KEYS = [k for k, _ in LABELS]
KO = dict(LABELS)
NUM_CLASSES = len(LABELS)


def label_from_filename(name: str):
    """For one-move-per-video data: the file name starts with the label,
    e.g. selca_minseok_01.mp4. Longest matching key wins."""
    stem = name.rsplit("/", 1)[-1]
    best = None
    for k in KEYS:
        if stem.startswith(k + "_") or stem.split(".")[0] == k:
            if best is None or len(k) > len(best):
                best = k
    return best
