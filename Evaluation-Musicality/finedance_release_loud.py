"""Loudness <-> body energy on the FineDance songs outside the test set: real dance only (MACE was trained on them).

Replicates the human side of the test-set result (Delta_real = S_own - S_matched_wrong > 0) on many more songs.

Pool (fixed before running, 2026-10-27):
  * public FineDance release songs (finedance_common.py) not in MACE's 19 test songs,
  * minus songs whose music duplicates a test song (duet twins 108 131 133 136 142 145, which MACE trained on),
  * minus songs whose motion and music lengths differ by more than 1 s (sync unclear),
  * at least one full 1020-frame slice,
  * minus songs whose audio is silent (RMS < 1e-6) in more than half the frames (only 169 'TheMotto', a broken file).
Units: music pieces. Songs with identical music (duets: the same track danced by two people) form one piece;
their dancers are averaged and a piece is never its own wrong song. Duplicates: same recording in the first 30 s,
allowing up to 1 s of offset (see music_groups).

Per slice (1020 frames every 510, as on the test set): music = loudness of the release wav (full song, then sliced),
motion = release joints. Wrong songs: 3 other pieces by tempo-first matching (robustness_loud.matches, full-song BPM),
one seeded-random slice of each. Variants: robustness_loud.VARIANTS with genre-first matching replaced by 3 random
pieces of any tempo.
Stats on pieces: Delta_real mean, bootstrap 95 % CI, one-sided Wilcoxon and sign-flip permutation (Delta > 0),
count of pieces with Delta > 0; per-genre breakdown for the primary pipeline.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/finedance_release_loud.py
"""
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from model.metric import SMPLSkeleton  # noqa: E402
import audio_features as af  # noqa: E402
import metrics as M  # noqa: E402
from multicomponent import SEED, smooth, boot_ci, wil, signflip_p  # noqa: E402
from robustness_loud import PRIMARY, VARIANTS, loudness_signals, motion_signals, bpm_from_wav, matches  # noqa: E402
from finedance_common import RELEASE, SLICE_LEN, SLICE_STRIDE, release_joints  # noqa: E402

OUT = os.path.join(HERE, 'results', 'finedance_release_loud.json')
TEST_SONGS = ['012', '036', '037', '063', '065', '092', '098', '109', '130', '132', '137', '143', '144', '161',
              '179', '193', '198', '204', '211']
MAX_LEN_DIFF = 1.0
DUP_R, DUP_ENV_R, DUP_MAX_LAG, DUP_SECONDS, DUP_SR = 0.9, 0.9, 1.0, 30, 8000
REL_VARIANTS = {('match_random' if k == 'match_genre_first' else k): ({'match': 'random'} if k == 'match_genre_first'
                                                                       else v) for k, v in VARIANTS.items()}


def wav(song):
    return os.path.join(RELEASE, 'music_wav', song + '.wav')


def label(song):
    lab = json.load(open(os.path.join(RELEASE, 'label_json', song + '.json')))
    return {**lab, 'style2': lab['style2'][0].upper() + lab['style2'][1:]}         # 'jazz' -> 'Jazz'


def music_groups(songs, path=None):
    """Union songs whose first DUP_SECONDS of audio are the same recording, allowing an offset of up to DUP_MAX_LAG s
    (096/165 'GoCrazy' are the same track shifted by 0.03 s; Motorica gTP 014/015 by 0.19 s).
    Shortlist: max 4 Hz loudness-envelope r within +-DUP_MAX_LAG s > DUP_ENV_R;
    confirm: max normalised waveform cross-correlation within +-DUP_MAX_LAG s > DUP_R. Returns {song: piece id}."""
    path = path or wav
    wave, env = [], []
    for s in songs:
        y, _ = librosa.load(path(s), sr=DUP_SR, duration=DUP_SECONDS)
        y = np.pad(y, (0, DUP_SR * DUP_SECONDS - len(y)))
        wave.append((y - y.mean()) / (y.std() + 1e-9))
        e = librosa.feature.rms(y=y, frame_length=DUP_SR // 2, hop_length=DUP_SR // 4)[0]
        env.append((e - e.mean()) / (e.std() + 1e-9))
    env = np.stack(env)
    k = int(DUP_MAX_LAG * 4)                        # envelope frames are 0.25 s: search the same +-DUP_MAX_LAG
    E = np.max([env[:, max(s, 0):env.shape[1] + min(s, 0)] @ env[:, max(-s, 0):env.shape[1] + min(-s, 0)].T
                / (env.shape[1] - abs(s)) for s in range(-k, k + 1)], axis=0)
    L = int(DUP_MAX_LAG * DUP_SR)

    def same_recording(a, b):
        c = fftconvolve(a, b[::-1], mode='full')[len(b) - 1 - L:len(b) + L]
        return c.max() / len(a) > DUP_R
    parent = list(range(len(songs)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(songs)):
        for j in range(i + 1, len(songs)):
            if E[i, j] > DUP_ENV_R and same_recording(wave[i], wave[j]):
                parent[root(j)] = root(i)
    return {s: songs[root(i)] for i, s in enumerate(songs)}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)
    all_songs = sorted({f[:-4] for f in os.listdir(os.path.join(RELEASE, 'motion'))}
                       & {f[:-4] for f in os.listdir(os.path.join(RELEASE, 'music_wav'))}
                       & {f[:-5] for f in os.listdir(os.path.join(RELEASE, 'label_json'))})
    group_all = music_groups(all_songs)
    test_pieces = {group_all[s] for s in TEST_SONGS}
    excluded = {'test': [], 'duplicates_test_music': [], 'length_mismatch': [], 'too_short': [], 'silent_music': []}
    pool = []
    for s in all_songs:
        n = len(np.load(os.path.join(RELEASE, 'motion', s + '.npy'), allow_pickle=False, mmap_mode='r'))
        if s in TEST_SONGS:
            excluded['test'].append(s)
        elif group_all[s] in test_pieces:
            excluded['duplicates_test_music'].append(s)
        elif abs(n / 30 - sf.info(wav(s)).duration) > MAX_LEN_DIFF:
            excluded['length_mismatch'].append(s)
        elif n < SLICE_LEN:
            excluded['too_short'].append(s)
        elif np.mean(af.rms_envelope(af.load_audio(wav(s))) < 1e-6) > 0.5:
            excluded['silent_music'].append(s)
        else:
            pool.append(s)
    piece_of = {s: group_all[s] for s in pool}
    pieces = sorted(set(piece_of.values()))
    genre = {p: label(p)['style2'] for p in pieces}
    print(f'{len(pool)} songs in {len(pieces)} pieces; excluded', {k: len(v) for k, v in excluded.items()})

    # slices: (song, start frame); music per piece (identical audio within a piece, use the piece's own song)
    slices = []
    music, real = {}, []
    for s in pool:
        j = release_joints(s, smpl, device)
        if piece_of[s] not in music:
            music[piece_of[s]] = loudness_signals(af.load_audio(wav(piece_of[s])))
        n_music = len(music[piece_of[s]]['rms'])
        for o in range(0, min(len(j), n_music) - SLICE_LEN + 1, SLICE_STRIDE):
            slices.append((s, o))
            real.append(motion_signals(j[o:o + SLICE_LEN]))
    slice_piece = np.array([piece_of[s] for s, _ in slices])
    print(len(slices), 'slices')

    bpm = {p: bpm_from_wav(wav(p)) for p in pieces}
    mt = {'tempo_first': matches(pieces, genre, bpm, 'tempo_first')}
    rng = np.random.default_rng(SEED)
    mt['random'] = {p: [str(x) for x in rng.choice([q for q in pieces if q != p], 3, replace=False)] for p in pieces}
    # wrong slice offsets: one seeded-random slice per matched piece (offsets valid for that piece's music and dances)
    piece_offsets = {p: sorted({o for (s, o), q in zip(slices, slice_piece) if q == p}) for p in pieces}
    pairing = {}
    for how, m in mt.items():
        rng = np.random.default_rng(SEED)
        pairing[how] = [[(w, int(rng.choice(piece_offsets[w]))) for w in m[piece_of[s]]] for s, _ in slices]

    results = {}
    for vname, change in REL_VARIANTS.items():
        cfg = {**PRIMARY, **change}
        sig, lag = cfg['sigma'], cfg['lag']
        ms = lambda p, o: smooth(music[p][cfg['music']][o:o + SLICE_LEN], sig)  # noqa: E731
        own, other = [], []
        for i, (s, o) in enumerate(slices):
            d = smooth(real[i][cfg['motion']], sig)
            own.append(M.lagged_corr(ms(piece_of[s], o), d, lag)[1])
            other.append(np.mean([M.lagged_corr(ms(w, wo), d, lag)[1] for w, wo in pairing[cfg['match']][i]]))
        own, other = np.array(own), np.array(other)
        ro = np.array([own[slice_piece == p].mean() for p in pieces])
        rx = np.array([other[slice_piece == p].mean() for p in pieces])
        dr = ro - rx
        rng = np.random.default_rng(SEED)
        st = {'real_own': float(ro.mean()), 'real_other': float(rx.mean()), 'd_real': float(dr.mean()),
              'd_real_ci': boot_ci(dr, rng), 'H1_wilcoxon_p': wil(dr, 'greater'),
              'H1_perm_p': signflip_p(dr, rng, 'greater'), 'n_real_pos': int((dr > 0).sum()), 'n': len(dr),
              'config': cfg}
        if vname == 'primary':
            st['per_piece'] = {p: {'real_own': float(a), 'real_other': float(b), 'genre': genre[p],
                                   'songs': [s for s in pool if piece_of[s] == p]} for p, a, b in zip(pieces, ro, rx)}
            st['per_genre'] = {}
            for g in sorted(set(genre.values())):
                sel = np.array([genre[p] == g for p in pieces])
                st['per_genre'][g] = {'n': int(sel.sum()), 'd_real': float(dr[sel].mean()),
                                      'n_pos': int((dr[sel] > 0).sum())}
        results[vname] = st
        print(f"{vname:18s} R own {st['real_own']:.3f} other {st['real_other']:.3f} dR {st['d_real']:+.3f} "
              f"[{st['d_real_ci'][0]:+.3f},{st['d_real_ci'][1]:+.3f}] p perm {st['H1_perm_p']:.4f} "
              f"wil {st['H1_wilcoxon_p']:.4f} | {st['n_real_pos']}/{st['n']} pieces > 0")

    out = {'results': results, 'pool': pool, 'excluded': excluded,
           'pieces': {p: [s for s in pool if piece_of[s] == p] for p in pieces},
           'duplicate_groups_all_songs': {g: [s for s in all_songs if group_all[s] == g]
                                          for g in sorted(set(group_all.values()))
                                          if sum(group_all[s] == g for s in all_songs) > 1},
           'bpm': bpm, 'genre': genre, 'matching': mt, 'n_slices': len(slices)}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print('per genre (primary):', json.dumps(results['primary']['per_genre']))
    print('saved', OUT)


if __name__ == '__main__':
    main()
