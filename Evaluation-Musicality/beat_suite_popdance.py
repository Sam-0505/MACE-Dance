"""BAS / event PLV / BPS of REAL dance on PopDanceSet against a tempo-matched null of other videos' beats (beat_suite.py).

Clips >= 16 s (as validity_real.py), first 1020 frames, global HybrIK joints; group = video (the null never uses the same
video); beats = librosa column 34 of the clip's audio.  Nulls as beat_suite --tempo_null: 'other' = 12 random clips of other
videos, 'tempo' = the 12 clips of other videos nearest in tempo (line-fit BPM).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/beat_suite_popdance.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import beat_suite as B  # noqa: E402
import validity_real as V  # noqa: E402


def main():
    units, vid = V.load_popdance(16.0, False)
    names = sorted(units)
    group = [vid[n] for n in names]
    beats = [units[n]['beats'] for n in names]
    sets = {'real': [[units[n]['real'][0].astype(np.float32)] for n in names]}
    bpm = np.array([60 * B.FPS / np.polyfit(np.arange(len(b)), b, 1)[0] if len(b) > 2 else np.nan for b in beats])
    nulls = {'other': None, 'tempo': ('tempo', bpm)}
    print('popdance', len(group), 'clips,', len(set(group)), 'videos', flush=True)
    res, tests, ng = B.evaluate(group, beats, sets, nulls)
    with open(os.path.join(HERE, 'results', 'beat_suite_popdance.json'), 'w') as f:
        json.dump({'n_groups': ng, 'table': res, 'tests': tests}, f, indent=1)
    print('saved')


if __name__ == '__main__':
    main()
