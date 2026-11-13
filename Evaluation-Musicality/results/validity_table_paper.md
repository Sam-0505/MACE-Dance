| Dataset | Motion source | Original | Mismatched | Δ | 95% CI | p | Advantage detected? |
|---|---|---:|---:|---:|---:|---:|:---:|
| **Beat-Phase Locking (BPL)** | | | | | | | |
| FineDance | Real Dance | 0.459 | 0.420 | +0.039 | [+0.013, +0.065] | 0.0063 | Yes |
|  | MACE-Dance | 0.304 | 0.305 | -0.001 | [-0.006, +0.004] | 0.6426 | No |
| Motorica | Real Dance | 0.420 | 0.384 | +0.037 | [+0.018, +0.055] | 0.0002 | Yes |
| AIST++ | Real Dance | 0.759 | 0.751 | +0.009 | [+0.003, +0.015] | 0.0024 | Yes |
| **Intensity–Motion Coupling (IMC)** | | | | | | | |
| FineDance | Real Dance | 0.219 | 0.092 | +0.127 | [+0.013, +0.229] | 0.0192 | Yes |
|  | MACE-Dance | 0.055 | 0.057 | -0.001 | [-0.040, +0.037] | 0.5278 | No |
| Motorica | Real Dance | 0.180 | 0.011 | +0.169 | [+0.125, +0.214] | 0.0001 | Yes |
| AIST++ | Real Dance | 0.027 | 0.072 | -0.045 | [-0.114, +0.024] | 0.8926 | No |

Original = own music, Mismatched = mean over 3 tempo-matched wrong songs, Δ = Original − Mismatched, p = one-sided sign-flip permutation test of Δ > 0 (10,000), Advantage detected = p < 0.05. Units: songs (FineDance 19), pieces (Motorica 52, AIST++ 70 = 60 basic + 10 advanced). BPL trim: 4 s each end; AIST++ pools the 60 basic dances (7–12 s, too short for 4 s: trimmed 1 s) and the 10 advanced ones (4 s). MACE-Dance rows: FineDance only (10 seeds), the dataset MACE was trained on; Motorica and AIST++ are out of its domain.
