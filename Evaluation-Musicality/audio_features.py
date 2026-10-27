"""EDGE-style `baseline` music features (35 dims at 30 fps), rebuilt so new audio
(ImperialDance clips, counterfactual edits) is described exactly like the
features the Motion Expert was trained and evaluated on.

Layout: [0] onset envelope | [1:21] MFCC | [21:33] chroma CENS | [33] peak one-hot | [34] beat one-hot
"""
import numpy as np
import librosa

FPS = 30
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH

ENVELOPE, PEAK, BEAT = 0, 33, 34


def estimate_tempo(y, sr):
    # EDGE falls back to librosa's global tempo estimate when the BPM is not encoded in the file name
    return float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr))[0])


def extract_baseline(y, start_bpm=None):
    """y: mono audio already at SR. Returns (T, 35) float32."""
    envelope = librosa.onset.onset_strength(y=y, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T

    peak_idxs = librosa.onset.onset_detect(onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0

    if start_bpm is None:
        start_bpm = estimate_tempo(librosa.resample(y, orig_sr=SR, target_sr=22050), 22050)
    _, beat_idxs = librosa.beat.beat_track(onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
                                           start_bpm=start_bpm, tightness=100)
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0

    return np.concatenate([envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
                          axis=-1).astype(np.float32)


def load_audio(path):
    y, _ = librosa.load(path, sr=SR)
    return y


def extract_file(path, n_frames=None, start_bpm=None):
    feat = extract_baseline(load_audio(path), start_bpm)
    return feat[:n_frames] if n_frames is not None else feat


def rms_envelope(y, n_frames=None):
    """Frame-wise RMS loudness on the same 30 fps grid as the baseline features (for EAS)."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    return rms[:n_frames] if n_frames is not None else rms
