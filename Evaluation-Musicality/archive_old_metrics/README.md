# Archived old metric scripts (2026-11-08)

Standalone scripts of the earlier Loud / beat-lock metrics. Nothing imports them. Moved with `git mv`, so history is kept
(`git mv archive_old_metrics/<file> .` restores one). Their results JSONs in `results/` are unchanged.

- `motorica_loudness.py`  old Loud on Motorica (squared energy, lag search)
- `aist_loudness.py`      old Loud on AIST++
- `beatlock_validate.py`  old soft-PLV check of the beat-locking loss

## Current metric scripts (run from Expert-Motion with .venv/bin/python)

- `loud_models.py`, `loud_models_motorica.py`   Loud: Gaussian sigma 15 QoM vs audio RMS, zero lag, own - wrong songs
- `lock_models.py`, `lock_metric_table.py`      Lock: continuous beat phase locking after Burger 2014 (whole-body QoM,
                                                15 % band-pass at the beat rate, Hilbert phase, 4 s trim), own-beat r
They still import `robustness_loud.py`, `multicomponent.py`, `finedance_release_loud.py`, `metrics.py` and `motorica_common.py`
(signal, matching and statistics helpers), so those stay in place.
