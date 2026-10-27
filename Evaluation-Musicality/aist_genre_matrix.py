"""Genre x genre matrix for loudness <-> movement energy on AIST++ (real dance only).

Question: is each genre's loudness character matched by its movement-energy character?
Clip-level descriptors, not time-aligned curves, so dances from different songs can be compared.

Data (facts checked 2026-10-27):
  * AIST++ public motions (411 of 1408 sequences) converted by aist_convert.py; ignore_list excluded.
  * Basic dances only (sBM): every one is exactly 16 beats at its piece's nominal tempo, so clip lengths
    are comparable. Advanced dances (64 beats, 1-4 per genre) are left out.
  * Sync: the refined AIST videos' soundtracks match wav[0:dur] at offset 0 (r >= 0.999, checked on 3 videos),
    so motion frame 0 = music time 0 and each dance uses the first len(motion)/60 s of its piece.
  * Every genre uses the same tempi (80-130 BPM; house 110-135), so tempo does not separate genres.
  * Joints: SMPLSkeleton FK with root = smpl_trans / smpl_scaling (SMPL scale, ~metres), 60 -> 30 fps.

Descriptors (each curve smoothed with Gaussian sigma 15 frames = 0.5 s at 30 fps):
  music   RMS in dB (15360 Hz, hop 512, top_db 80): mean, std, p95 - p5
  motion  log10 mean squared joint speed:            mean, std, p95 - p5

Mapping: ridge regression (alpha 1) from z-scored music to z-scored motion descriptors (z over all clips),
fitted leave-one-genre-out: clips of genre A are predicted by a model that never saw genre A.

Matrix: cell (A, B) = Euclidean distance between the mean prediction from genre-A music and the mean actual
motion descriptors of genre B. Lower = closer; a matched genre should be lowest on the diagonal.
Test: mean diagonal distance vs N_PERM random permutations of the column genres (keeps row/column effects).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/aist_genre_matrix.py
"""
import glob
import json
import os
import sys

import librosa
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from vis import SMPLSkeleton  # noqa: E402
import audio_features as af  # noqa: E402
import metrics as M  # noqa: E402

DATA = os.path.join(HERE, 'data', 'aist')
OUT = os.path.join(HERE, 'results', 'aist_genre_matrix.json')
FIG_DIR = os.path.join(HERE, 'results', 'figures')
SIGMA = 15
ALPHA = 1.0
N_PERM = 100000
SEED = 0
GENRES = ['gBR', 'gPO', 'gLO', 'gWA', 'gMH', 'gLH', 'gHO', 'gKR', 'gJS', 'gJB']
GENRE_NAMES = {'gBR': 'Break', 'gPO': 'Pop', 'gLO': 'Lock', 'gWA': 'Waack', 'gMH': 'Middle Hip-hop',
               'gLH': 'LA Hip-hop', 'gHO': 'House', 'gKR': 'Krump', 'gJS': 'Street Jazz', 'gJB': 'Ballet Jazz'}
DESC = ['mean', 'std', 'range']
LABEL_OFFSET = {'gBR': (-6, 6, 'right'), 'gMH': (6, 6, 'left'), 'gLH': (6, -15, 'left'), 'gWA': (-6, -12, 'right'),
                'gHO': (0, 8, 'center')}   # scatter labels that would otherwise overlap


def describe(x):
    x = gaussian_filter1d(np.asarray(x, dtype=np.float64), SIGMA)
    return [x.mean(), x.std(), np.percentile(x, 95) - np.percentile(x, 5)]


def joints(path, skel, device):
    d = np.load(path)
    poses = torch.from_numpy(d['smpl_poses'][::2]).to(device).reshape(1, -1, 24, 3)
    root = torch.from_numpy(d['smpl_trans'][::2] / d['smpl_scaling']).to(device)[None]
    return skel.forward(poses, root).squeeze(0).cpu().numpy()


def ridge_fit(X, Y, alpha=ALPHA):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    reg = alpha * np.eye(Xb.shape[1])
    reg[-1, -1] = 0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)


def ridge_predict(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skel = SMPLSkeleton(device)
    ignore = set(open(os.path.join(DATA, 'ignore_list.txt')).read().split())
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(DATA, 'motions_npz', '*_sBM_*.npz')))
    names = [n for n in names if n not in ignore]

    rms_db = {}
    music_X, motion_Y, genre = [], [], []
    for n in names:
        piece = n.split('_')[4]
        if piece not in rms_db:
            y = af.load_audio(os.path.join(DATA, 'music', piece + '.wav'))
            rms_db[piece] = librosa.amplitude_to_db(af.rms_envelope(y), ref=1.0, top_db=80)
        j = joints(os.path.join(DATA, 'motions_npz', n + '.npz'), skel, device)
        T = len(j)
        assert len(rms_db[piece]) >= T, (n, len(rms_db[piece]), T)
        music_X.append(describe(rms_db[piece][:T]))
        motion_Y.append(describe(np.log10(M.motion_energy(j) + 1e-10)))
        genre.append(n[:3])
    X, Y, genre = np.array(music_X), np.array(motion_Y), np.array(genre)
    print(f'{len(names)} basic dances, per genre:', {g: int((genre == g).sum()) for g in GENRES})

    Xz = (X - X.mean(0)) / X.std(0)
    Yz = (Y - Y.mean(0)) / Y.std(0)

    # leave-one-genre-out predictions
    P = np.zeros_like(Yz)
    for g in GENRES:
        tr = genre != g
        P[~tr] = ridge_predict(ridge_fit(Xz[tr], Yz[tr]), Xz[~tr])
    cv_r2 = 1 - ((Yz - P) ** 2).sum(0) / ((Yz - Yz.mean(0)) ** 2).sum(0)
    W_all = ridge_fit(Xz, Yz)

    pred_mean = np.array([P[genre == g].mean(0) for g in GENRES])
    act_mean = np.array([Yz[genre == g].mean(0) for g in GENRES])
    mus_mean = np.array([Xz[genre == g].mean(0) for g in GENRES])
    D = np.linalg.norm(pred_mean[:, None, :] - act_mean[None, :, :], axis=2)   # rows: music genre, cols: motion

    rng = np.random.default_rng(SEED)
    k = len(GENRES)
    obs = np.trace(D) / k
    null = np.array([D[np.arange(k), rng.permutation(k)].mean() for _ in range(N_PERM)])
    p = float((np.sum(null <= obs) + 1) / (N_PERM + 1))
    row_top1 = int(sum(np.argmin(D[i]) == i for i in range(k)))
    col_top1 = int(sum(np.argmin(D[:, i]) == i for i in range(k)))
    rho_level = spearmanr(mus_mean[:, 0], act_mean[:, 0])

    out = {
        'settings': {'sigma': SIGMA, 'alpha': ALPHA, 'n_perm': N_PERM, 'seed': SEED, 'descriptors': DESC,
                     'music': 'RMS dB', 'motion': 'log10 mean squared joint speed', 'fps': 30,
                     'cv': 'leave-one-genre-out', 'units': 'basic (sBM) dances, ignore list excluded'},
        'n_per_genre': {g: int((genre == g).sum()) for g in GENRES},
        'genres': GENRES, 'genre_names': GENRE_NAMES,
        'matrix_distance': D.round(4).tolist(),
        'diag_mean': float(obs), 'offdiag_mean': float(D[~np.eye(k, dtype=bool)].mean()),
        'perm_null_mean': float(null.mean()), 'perm_p': p,
        'row_top1': row_top1, 'col_top1': col_top1,
        'cv_r2_per_motion_descriptor': dict(zip(DESC, cv_r2.round(4).tolist())),
        'ridge_weights_all_data': {'rows': DESC + ['bias'], 'cols': DESC, 'W': W_all.round(4).tolist()},
        'genre_music_mean_z': dict(zip(GENRES, mus_mean.round(4).tolist())),
        'genre_motion_mean_z': dict(zip(GENRES, act_mean.round(4).tolist())),
        'genre_level_spearman': {'rho': float(rho_level.statistic), 'p': float(rho_level.pvalue),
                                 'what': 'genre mean loudness level vs genre mean energy level (10 genres)'},
        'genre_raw_means': {g: {'music_db_mean': float(X[genre == g, 0].mean()),
                                'motion_log_energy_mean': float(Y[genre == g, 0].mean())} for g in GENRES},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ('diag_mean', 'offdiag_mean', 'perm_null_mean', 'perm_p', 'row_top1',
                                          'col_top1', 'cv_r2_per_motion_descriptor', 'genre_level_spearman')},
                     indent=1))
    make_figures(out, X, Y, genre)


def make_figures(out, X, Y, genre):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    surface, ink, ink2 = '#fcfcfb', '#0b0b0b', '#52514e'
    seq = ['#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb']   # dark = close
    cmap = LinearSegmentedColormap.from_list('closeness', seq)
    labels = [GENRE_NAMES[g] for g in GENRES]
    D = np.array(out['matrix_distance'])
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 7), facecolor=surface)
    ax.set_facecolor(surface)
    vmax = np.percentile(D, 90)                     # colour capped so one outlier row doesn't flatten the rest
    im = ax.imshow(D, cmap=cmap, vmax=vmax)
    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            dark = (min(D[i, j], vmax) - D.min()) / (vmax - D.min()) < 0.5
            ax.text(j, i, f'{D[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='#ffffff' if dark else ink, fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(len(GENRES)), labels, rotation=40, ha='right', color=ink2)
    ax.set_yticks(range(len(GENRES)), labels, color=ink2)
    ax.set_xlabel('Motion genre (actual movement-energy descriptors)', color=ink)
    ax.set_ylabel('Music genre (movement predicted from loudness)', color=ink)
    ax.set_title(f"Loudness → movement energy, AIST++ basic dances\n"
                 f"diagonal {out['diag_mean']:.2f} vs shuffled {out['perm_null_mean']:.2f}, "
                 f"p = {out['perm_p']:.3f}; lower = closer", color=ink, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(f'distance (darker = closer; colour capped at {vmax:.1f})', color=ink2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'aist_genre_matrix.png'), dpi=160, facecolor=surface)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.2), facecolor=surface)
    ax.set_facecolor(surface)
    for g in GENRES:
        m = genre == g
        ax.scatter(X[m, 0].mean(), Y[m, 0].mean(), s=64, color='#2a78d6', edgecolor=surface, linewidth=2, zorder=3)
        dx, dy, ha = LABEL_OFFSET.get(g, (6, 4, 'left'))
        ax.annotate(GENRE_NAMES[g], (X[m, 0].mean(), Y[m, 0].mean()), xytext=(dx, dy), textcoords='offset points',
                    fontsize=9, color=ink, ha=ha)
    rho = out['genre_level_spearman']
    ax.set_title(f"Genre means: loudness level vs movement-energy level (Spearman ρ = {rho['rho']:.2f}, "
                 f"p = {rho['p']:.2f})", color=ink, fontsize=10)
    ax.set_xlabel('Mean music loudness (dB RMS)', color=ink2)
    ax.set_ylabel('Mean log10 movement energy', color=ink2)
    ax.grid(color='#e6e5e1', linewidth=0.8)
    ax.tick_params(colors=ink2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'aist_genre_scatter.png'), dpi=160, facecolor=surface)
    plt.close(fig)
    print('figures saved to', FIG_DIR)


if __name__ == '__main__':
    main()
