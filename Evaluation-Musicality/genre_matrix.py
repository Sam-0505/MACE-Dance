"""Genre x genre matrices on FineDance and Motorica (real dance only): does a genre's music character predict its
dance character?

Fixed before running (2026-10-27). Same method as aist_genre_matrix.py, on dances choreographed to (or improvised on)
their own music. One matrix per dataset: skeletons (24 vs 21 joints), capture and mastering differ, so descriptor
levels are not comparable across datasets.
FineDance: every real FineDance dance in the public release that passes the finedance_release_loud.py pool rules,
  test songs included (this matrix uses no MACE output): the 150 non-test pieces plus the 19 test songs, the test
  songs' duet twins merged into their piece. Motion = release joints (clean). Units = music pieces (dancers averaged).
Motorica: the motorica_loudness.py pool (62 takes, 52 pieces), 21 BVH body joints; units = music pieces.
Descriptors per 34 s slice (1020 frames every 510), each signal smoothed with sigma 15, summarised by mean, std and
  p95 - p5, then averaged over a piece's slices and dancers:
  music   loudness dB (RMS, top_db 80), percussive dB (RMS of the HPSS percussive part), log10 treble flux
          (6.4-12.8 kHz)
  motion  log10 motion energy, log10 upper-body acceleration, log10 travel-free leg + head/hand speed
Matrix: genres with >= 7 pieces on FineDance, >= 3 on Motorica (all styles but Locking, which has 1). Ridge (alpha 1) from z-scored music to z-scored motion descriptors (z over all
  pieces), fitted leave-one-genre-out on all pieces outside the row genre (small genres train only). Cell (A, B) =
  distance between the mean prediction from genre-A music and genre B's mean actual motion. Test: mean diagonal vs
  100,000 column permutations; also rows/columns whose own genre is closest and the cross-validated R^2 per motion
  descriptor.
Variant loudness_only: music = loudness dB descriptors, motion = log energy descriptors (the AIST++ setup).
FineDance top-level styles (added 2026-10-27, fixed before running): the same matrix with FineDance's own style1
  labels (Street, Classic, Folk, Mix; set by the dataset authors) in place of genres, all pieces, from the saved
  descriptors (--style1).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/genre_matrix.py [finedance|motorica ...]
    .venv/bin/python ../Evaluation-Musicality/genre_matrix.py --style1      # needs the finedance results
"""
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from model.metric import SMPLSkeleton  # noqa: E402
import audio_features as af  # noqa: E402
import metrics as M  # noqa: E402
from multicomponent import FLUX_SR, FLUX_HOP, FLUX_NFFT, HIGH_BAND, UPPER, band_flux, smooth  # noqa: E402
from finedance_common import RELEASE, SLICE_LEN, SLICE_STRIDE, release_joints  # noqa: E402
from finedance_release_loud import TEST_SONGS, MAX_LEN_DIFF, label  # noqa: E402
from aist_genre_matrix import ridge_fit, ridge_predict  # noqa: E402
import bass_treble as BT  # noqa: E402
import motorica_common as MC  # noqa: E402
from perc_distinct import UPPER_MOTORICA  # noqa: E402

OUT = os.path.join(HERE, 'results', 'genre_matrix.json')
FIG_DIR = os.path.join(HERE, 'results', 'figures')
SIGMA, N_PERM, SEED = 15, 100000, 0
MIN_PIECES = {'finedance': 7, 'motorica': 3}
MUSIC = ['loud_db', 'perc_db', 'treble_log']
MOTION = ['energy_log', 'uacc_log', 'body_log']
STATS = ['mean', 'std', 'range']
VARIANTS = {'primary': (MUSIC, MOTION), 'loudness_only': (['loud_db'], ['energy_log'])}


def describe(x):
    x = smooth(x, SIGMA)
    return [x.mean(), x.std(), np.percentile(x, 95) - np.percentile(x, 5)]


def full_music(path):
    y = af.load_audio(path)
    y48, _ = librosa.load(path, sr=FLUX_SR)
    S = np.abs(librosa.stft(y48, n_fft=FLUX_NFFT, hop_length=FLUX_HOP))
    db = lambda a: librosa.amplitude_to_db(a, ref=1.0, top_db=80)  # noqa: E731
    return {'loud_db': db(af.rms_envelope(y)),
            'perc_db': db(librosa.feature.rms(y=librosa.effects.percussive(y), hop_length=af.HOP_LENGTH)[0]),
            'treble_log': np.log10(band_flux(S, librosa.fft_frequencies(sr=FLUX_SR, n_fft=FLUX_NFFT), HIGH_BAND)
                                   + 1e-6)}


def motion_signals(j, src):
    J, (ax, _) = BT.JOINTS['motorica' if src == 'motorica' else 'smpl'], BT.UP[src]
    upper = UPPER_MOTORICA if src == 'motorica' else UPPER
    j = np.asarray(j, dtype=np.float64)
    tf = j.copy()
    horiz = [a for a in range(3) if a != ax]
    tf[:, :, horiz] -= j[:, J['pelvis']:J['pelvis'] + 1, horiz]
    idx = J['legs'] + J['hands']
    acc = np.linalg.norm(j[2:, upper] - 2 * j[1:-1, upper] + j[:-2, upper], axis=2).mean(1)
    return {'energy_log': np.log10(M.motion_energy(j) + 1e-10),
            'uacc_log': np.log10(np.pad(acc, (1, 1), mode='edge') + 1e-10),
            'body_log': np.log10(np.pad(np.linalg.norm(tf[1:, idx] - tf[:-1, idx], axis=2).mean(1), (0, 1),
                                        mode='edge') + 1e-10)}


def pieces_and_songs():
    """{piece: [songs]} for the non-test pool plus the test songs (with their duet twins), and the exclusions."""
    rel = json.load(open(os.path.join(HERE, 'results', 'finedance_release_loud.json')))
    pieces = {p: list(s) for p, s in rel['pieces'].items()}
    dup = rel['duplicate_groups_all_songs']
    excluded = {}
    for t in TEST_SONGS:
        group = next((g for g in dup.values() if t in g), [t])
        songs = []
        for s in group:
            if s not in TEST_SONGS and s not in rel['excluded']['duplicates_test_music']:
                continue
            n = len(np.load(os.path.join(RELEASE, 'motion', s + '.npy'), mmap_mode='r'))
            if abs(n / 30 - sf.info(os.path.join(RELEASE, 'music_wav', s + '.wav')).duration) > MAX_LEN_DIFF:
                excluded[s] = 'length_mismatch'
            elif n < SLICE_LEN:
                excluded[s] = 'too_short'
            else:
                songs.append(s)
        if songs:
            pieces.setdefault(t, sorted(set(pieces.get(t, [])) | set(songs)))
    return pieces, excluded


def matrix(X, Y, genre, genres):
    Xz, Yz = (X - X.mean(0)) / X.std(0), (Y - Y.mean(0)) / Y.std(0)
    P = np.zeros_like(Yz)
    for g in set(genre):
        tr = genre != g
        P[~tr] = ridge_predict(ridge_fit(Xz[tr], Yz[tr]), Xz[~tr])
    cv_r2 = 1 - ((Yz - P) ** 2).sum(0) / ((Yz - Yz.mean(0)) ** 2).sum(0)
    pred = np.array([P[genre == g].mean(0) for g in genres])
    act = np.array([Yz[genre == g].mean(0) for g in genres])
    D = np.linalg.norm(pred[:, None, :] - act[None, :, :], axis=2)          # rows: music genre, cols: motion genre
    k = len(genres)
    rng = np.random.default_rng(SEED)
    obs = np.trace(D) / k
    null = np.array([D[np.arange(k), rng.permutation(k)].mean() for _ in range(N_PERM)])
    return {'matrix_distance': D.round(4).tolist(), 'diag_mean': float(obs),
            'offdiag_mean': float(D[~np.eye(k, dtype=bool)].mean()), 'perm_null_mean': float(null.mean()),
            'perm_p': float((np.sum(null <= obs) + 1) / (N_PERM + 1)),
            'row_top1': int(sum(np.argmin(D[i]) == i for i in range(k))),
            'col_top1': int(sum(np.argmin(D[:, i]) == i for i in range(k))),
            'diag_rank_in_row': [int((D[i] < D[i, i]).sum()) + 1 for i in range(k)],
            'cv_r2': cv_r2.round(4).tolist()}


def piece_descriptors(pieces, music_path, joints_of, src):
    """{piece: {signal: [mean, std, range]}} averaged over the piece's slices and dancers."""
    desc = {}
    for p in sorted(pieces):
        music = full_music(music_path(p))
        rows = []
        for s in pieces[p]:
            j = joints_of(s)
            for o in range(0, min(len(j), len(music['loud_db'])) - SLICE_LEN + 1, SLICE_STRIDE):
                mot = motion_signals(j[o:o + SLICE_LEN], src)
                rows.append({**{k: describe(music[k][o:o + SLICE_LEN]) for k in MUSIC},
                             **{k: describe(mot[k]) for k in MOTION}})
        assert rows, p
        desc[p] = {k: np.mean([r[k] for r in rows], axis=0).tolist() for k in MUSIC + MOTION}
        desc[p]['n_slices'] = len(rows)
    return desc


def build(desc, genre_of, genres, settings, extra):
    names = sorted(desc)
    genre = np.array([genre_of[p] for p in names])
    out = {'settings': settings, 'genres': genres, 'n_per_genre': {g: int((genre == g).sum()) for g in sorted(set(genre))},
           'descriptors': desc, 'genre': dict(zip(names, genre.tolist())), 'variants': {}, **extra}
    for vname, (mk, dk) in VARIANTS.items():
        X = np.array([[v for k in mk for v in desc[p][k]] for p in names])
        Y = np.array([[v for k in dk for v in desc[p][k]] for p in names])
        res = matrix(X, Y, genre, genres)
        res['music'], res['motion'] = mk, dk
        res['cv_r2'] = dict(zip([f'{k}_{s}' for k in dk for s in STATS], res['cv_r2']))
        out['variants'][vname] = res
        print(' ', vname, json.dumps({k: res[k] for k in ('diag_mean', 'offdiag_mean', 'perm_null_mean', 'perm_p',
                                                           'row_top1', 'col_top1', 'diag_rank_in_row', 'cv_r2')}))
    return out


def run_style1(fd):
    """FineDance matrix over the dataset's own top-level styles, from the saved per-piece descriptors."""
    style1 = {p: label(p)['style1'] for p in fd['descriptors']}
    counts = {g: sum(v == g for v in style1.values()) for g in set(style1.values())}
    genres = sorted(counts, key=lambda g: -counts[g])
    print(f'finedance style1: {counts}', flush=True)
    return build(fd['descriptors'], style1, genres, {**fd['settings'], 'grouping': 'style1', 'min_pieces': 1},
                 {'pieces': fd['pieces'], 'excluded': fd['excluded']})


def run(dataset, smpl, device):
    if dataset == 'finedance':
        pieces, excluded = pieces_and_songs()
        genre_of = {p: label(p)['style2'] for p in pieces}
        desc = piece_descriptors(pieces, lambda p: os.path.join(RELEASE, 'music_wav', p + '.wav'),
                                 lambda s: release_joints(s, smpl, device), 'release')
    else:
        mot = json.load(open(os.path.join(HERE, 'results', 'motorica_loudness.json')))
        pieces, excluded, genre_of = mot['pieces'], {}, mot['style']
        desc = piece_descriptors(pieces, MC.wav_path, MC.body_joints, 'motorica')
    counts = {g: sum(v == g for p, v in genre_of.items() if p in pieces) for g in set(genre_of[p] for p in pieces)}
    genres = sorted([g for g, n in counts.items() if n >= MIN_PIECES[dataset]], key=lambda g: -counts[g])
    print(f'{dataset}: {len(pieces)} pieces; per genre {counts}; matrix genres {genres}; excluded {excluded}',
          flush=True)
    settings = {'sigma': SIGMA, 'min_pieces': MIN_PIECES[dataset], 'n_perm': N_PERM, 'seed': SEED, 'stats': STATS}
    return build(desc, {p: genre_of[p] for p in pieces}, genres, settings, {'pieces': pieces, 'excluded': excluded})


LABELS = {'gLH': 'Hip-hop', 'gPO': 'Popping', 'gCH': 'Charleston', 'gKR': 'Krump', 'gJZ': 'Jazz', 'gTP': 'Tap',
          'gLO': 'Locking'}
TITLES = {'finedance': 'FineDance', 'motorica': 'Motorica', 'finedance_style1': 'FineDance top-level styles'}


def make_figures(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    surface, ink, ink2, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1'
    cmap = LinearSegmentedColormap.from_list('closeness', ['#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec',
                                                           '#9ec5f4', '#cde2fb'])          # dark = close
    os.makedirs(FIG_DIR, exist_ok=True)
    for ds, r in results.items():
        v = r['variants']['primary']
        D, labels = np.array(v['matrix_distance']), [LABELS.get(g, g) for g in r['genres']]
        labels = [f"{l} ({r['n_per_genre'][g]})" for l, g in zip(labels, r['genres'])]
        k = len(labels)
        fig, ax = plt.subplots(figsize=(1.0 * k + 2.6, 0.85 * k + 1.9), facecolor=surface)
        ax.set_facecolor(surface)
        vmax = np.percentile(D, 90)
        im = ax.imshow(D, cmap=cmap, vmax=vmax)
        for i in range(k):
            for j in range(k):
                dark = (min(D[i, j], vmax) - D.min()) / (vmax - D.min()) < 0.5
                ax.text(j, i, f'{D[i, j]:.2f}', ha='center', va='center', fontsize=8,
                        color='#ffffff' if dark else ink, fontweight='bold' if i == j else 'normal')
        ax.set_xticks(range(k), labels, rotation=40, ha='right', color=ink2)
        ax.set_yticks(range(k), labels, color=ink2)
        ax.set_xlabel('Dance genre (actual motion descriptors)', color=ink)
        ax.set_ylabel('Music genre (motion predicted from its music)', color=ink)
        ax.set_title(f"{TITLES[ds]}: music character → dance character\n"
                     f"diagonal {v['diag_mean']:.2f} vs shuffled {v['perm_null_mean']:.2f}, p = {v['perm_p']:.3f}; "
                     f"lower = closer", color=ink, fontsize=11)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(f'distance (darker = closer; colour capped at {vmax:.1f})', color=ink2)
        cb.outline.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f'genre_matrix_{ds}.png'), dpi=160, facecolor=surface)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.8), facecolor=surface, squeeze=False)
    for ax, (ds, r) in zip(axes[0], results.items()):
        ax.set_facecolor(surface)
        d, gen = r['descriptors'], r['genre']
        for g in r['genres']:
            ps = [p for p in d if gen[p] == g]
            x = np.mean([d[p]['perc_db'][0] for p in ps])
            y = np.mean([d[p]['energy_log'][0] for p in ps])
            ax.scatter(x, y, s=20 + 6 * len(ps), color='#2a78d6', edgecolor=surface, linewidth=2, zorder=3)
            ax.annotate(LABELS.get(g, g), (x, y), xytext=(7, 3), textcoords='offset points', fontsize=9, color=ink)
        ax.set_title(f'{TITLES[ds]}: genre means', color=ink, fontsize=11)
        ax.set_xlabel('Mean percussive energy of the music (dB)', color=ink2)
        ax.set_ylabel('Mean log10 movement energy', color=ink2)
        ax.grid(color=grid, linewidth=0.8)
        ax.tick_params(colors=ink2)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'genre_means_scatter.png'), dpi=160, facecolor=surface)
    plt.close(fig)
    print('figures saved to', FIG_DIR)


def main():
    if '--style1' in sys.argv:
        results = json.load(open(OUT))
        results['finedance_style1'] = run_style1(results['finedance'])
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=1)
        make_figures(results)
        return
    if '--figures' in sys.argv:
        make_figures(json.load(open(OUT)))
        return
    which = [a for a in sys.argv[1:] if a in ('finedance', 'motorica')] or ['finedance', 'motorica']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for d in which:
        results[d] = run(d, smpl, device)
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=1)
    print('saved', OUT)
    make_figures(results)


if __name__ == '__main__':
    main()
