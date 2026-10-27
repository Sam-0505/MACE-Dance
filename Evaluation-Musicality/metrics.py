"""Music-motion alignment metrics on 3D joints sampled at 30 fps.

All functions take one clip: joints (T, J, 3) plus per-frame music signals on the
same 30 fps grid (see audio_features.py). Times are in frames unless noted.
"""
import numpy as np
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

FPS = 30
SIGMA_FRAMES = 3                    # sqrt(9) from the authors' BA; 0.1 s at 30 fps
TAU_FRAMES = 3                      # ABHS hit window (0.1 s); TMD-Bench gives no value
ACCENT_SMOOTH = 2                   # Gaussian sigma (frames) for frame-level accent curves
ENERGY_SMOOTH = 15                  # Gaussian sigma (frames) for slower dynamics (0.5 s)
MAX_LAG = 15                        # +/- 0.5 s search window for lagged correlations


# ---------------------------------------------------------------------------
# Authors' BAS (Expert-Motion/model/metric.py), kept identical except that
# get_db no longer hard-codes 24 joints so it also runs on ImperialDance.
# ---------------------------------------------------------------------------
def BA(music_beats, motion_beats):
    if len(music_beats) == 0:
        return 0
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats[0] - bb) ** 2) / 2 / 9)
    return (ba / len(music_beats))


def get_mb(music_beats):
    t = music_beats.shape[0]
    beats = music_beats.astype(bool)
    beat_axis = np.arange(t)
    beat_axis = beat_axis[beats]
    return beat_axis


def get_db(keypoints):
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats


def bas(beat_onehot, joints):
    return BA(get_mb(beat_onehot), get_db(joints))


# ---------------------------------------------------------------------------
# TMD-Bench VBCS / ABHS. Motion accents are local maxima of the same smoothed
# velocity curve whose minima the authors' BAS uses.
# ---------------------------------------------------------------------------
def motion_accents(joints):
    vel = np.mean(np.linalg.norm(joints[1:] - joints[:-1], axis=2), axis=1)
    return argrelextrema(G(vel, 5), np.greater)[0]


def vbcs(beat_onehot, joints, sigma=SIGMA_FRAMES):
    beats, accents = get_mb(beat_onehot), motion_accents(joints)
    if len(beats) == 0 or len(accents) == 0:
        return np.nan
    d = np.min(np.abs(accents[:, None] - beats[None, :]), axis=1)
    return float(np.mean(np.exp(-d ** 2 / (2 * sigma ** 2))))


def abhs(beat_onehot, joints, tau=TAU_FRAMES):
    beats, accents = get_mb(beat_onehot), motion_accents(joints)
    if len(beats) == 0 or len(accents) == 0:
        return np.nan
    d = np.min(np.abs(beats[:, None] - accents[None, :]), axis=1)
    return float(np.mean(d < tau))


# ---------------------------------------------------------------------------
# Proposed: accent-strength (AAS) and dynamics/energy (EAS) alignment.
# ---------------------------------------------------------------------------
def motion_acceleration(joints):
    """Mean joint acceleration magnitude, (T,) aligned to frames 1..T-2 and edge-padded."""
    acc = np.linalg.norm(joints[2:] - 2 * joints[1:-1] + joints[:-2], axis=2).mean(axis=1)
    return np.pad(acc, (1, 1), mode='edge')


def motion_energy(joints):
    """Mean squared joint speed, (T,) edge-padded."""
    v2 = (np.linalg.norm(joints[1:] - joints[:-1], axis=2) ** 2).mean(axis=1)
    return np.pad(v2, (0, 1), mode='edge')


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def lagged_corr(music, motion, max_lag=MAX_LAG):
    """Pearson r at zero lag, and the max r over lags in [-max_lag, max_lag].
    Positive lag = motion responds after the music."""
    T = min(len(music), len(motion))
    a, b = _z(music[:T]), _z(motion[:T])
    best, best_lag = -np.inf, 0
    r0 = float(np.mean(a * b))
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[:T - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:T + lag])
        r = np.corrcoef(x, y)[0, 1]
        if r > best:
            best, best_lag = r, lag
    return r0, float(best), best_lag


def aas(onset_envelope, joints):
    return lagged_corr(G(onset_envelope, ACCENT_SMOOTH), G(motion_acceleration(joints), ACCENT_SMOOTH))


def eas(rms, joints):
    return lagged_corr(G(rms, ENERGY_SMOOTH), G(motion_energy(joints), ENERGY_SMOOTH))


# ---------------------------------------------------------------------------
# Clip-level bundle + chance baseline from mismatched music/motion pairs.
# ---------------------------------------------------------------------------
def clip_scores(music_feat, rms, joints):
    """music_feat: (T, 35) baseline features; rms: (T,); joints: (T, J, 3)."""
    T = min(len(music_feat), len(rms), len(joints))
    music_feat, rms, joints = music_feat[:T], rms[:T], joints[:T]
    aas0, aas_max, aas_lag = aas(music_feat[:, 0], joints)
    eas0, eas_max, eas_lag = eas(rms, joints)
    return {
        'BAS': bas(music_feat[:, 34], joints),
        'VBCS': vbcs(music_feat[:, 34], joints),
        'ABHS': abhs(music_feat[:, 34], joints),
        'AAS': aas0, 'AAS_max': aas_max, 'AAS_lag': aas_lag,
        'EAS': eas0, 'EAS_max': eas_max, 'EAS_lag': eas_lag,
    }


def mismatched_scores(musics, rmss, joints_list, groups, seed=0):
    """Score each motion against music from a different group (e.g. another song) to get chance level."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(len(joints_list)):
        j = rng.choice(np.flatnonzero(groups != groups[i]))
        rows.append(clip_scores(musics[j], rmss[j], joints_list[i]))
    return rows


def summarize(rows):
    keys = rows[0].keys()
    return {k: (float(np.nanmean([r[k] for r in rows])), float(np.nanstd([r[k] for r in rows]))) for k in keys}


def bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray([x for x in values if np.isfinite(x)])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
