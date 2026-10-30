"""Session-aware internal development splits for speaker classification."""

import random
from collections import defaultdict
from pathlib import Path


def split_sessions(
    files: list[str], train_percent: float
) -> tuple[list[str], list[str]]:
    """Keep every identity in training and hold out whole recording sessions.

    The ratio is applied to sessions within each speaker; utterance counts may
    differ from that ratio. Single-session identities remain training-only.
    The caller owns the preparation RNG seed, independently of model seeds.
    """
    if not 0 < train_percent < 100:
        raise ValueError(
            "session splitting requires a training percentage between 0 and 100"
        )
    groups = defaultdict(lambda: defaultdict(list))
    for file in sorted(files):
        path = Path(file)
        groups[path.parent.parent.name][path.parent.name].append(file)
    train, dev = [], []
    for speaker in sorted(groups):
        sessions = sorted(groups[speaker])
        random.shuffle(sessions)
        heldout_count = (
            max(
                1,
                min(
                    len(sessions) - 1,
                    round(len(sessions) * (100 - train_percent) / 100),
                ),
            )
            if len(sessions) > 1
            else 0
        )
        heldout = set(sessions[:heldout_count])
        for session in sessions:
            (dev if session in heldout else train).extend(
                groups[speaker][session]
            )
    return train, dev
