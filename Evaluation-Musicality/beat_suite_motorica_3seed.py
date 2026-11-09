"""beat_suite on Motorica with the FIRST 3 SEEDS of every model (same number for all), incl. MEGADance.
beat_suite.py averages every seed on disk (released 10, mask 5, others 3); this wrapper truncates to 3 and adds MEGADance.
    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/beat_suite_motorica_3seed.py [--root_relative]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import beat_suite as B  # noqa: E402

N_SEEDS = 3
_load = B.load_motorica


def load():
    group, beats, sets, nulls = _load()
    sets = {k: ([c[:N_SEEDS] for c in v] if k != 'real' else v) for k, v in sets.items()}
    return group, beats, sets, nulls


B.load_motorica = load
MODELS = ['final=mace_local2s_guide_s0.003_foot10_frz', 'critic_v5=mace_local2s_criticv5_s0.003_foot10_frz',
          'mask_2s=mace_local2s', 'edge=edge', 'lodge=lodge', 'bailando=bailando', 'mega=mega']
sys.argv = [sys.argv[0], '--dataset', 'motorica', '--tag', '_3seed', '--tempo_null', '--mot_models', *MODELS,
            *sys.argv[1:]]
B.main()
