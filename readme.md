<p align="center">
  <h1 align="center">🎵 MACE-Dance</h1>
  <h3 align="center">Motion-Appearance Cascaded Experts for Music-Driven Dance Video Generation</h3>
<p align="center">
    Kaixing Yang<sup>1</sup> ·
    Jiashu Zhu<sup>2,*</sup> ·
    Xulong Tang<sup>5</sup> ·
    Ziqiao Peng<sup>1</sup> ·
    Xiangyue Zhang<sup>4</sup>
    <br>
    Puwei Wang<sup>1,†</sup> ·
    Jiahong Wu<sup>2,†</sup> ·
    Xiangxiang Chu<sup>2</sup> ·
    Hongyan Liu<sup>3,†</sup> ·
    Jun He<sup>1</sup>
    <br><br>
    <sup>1</sup>Renmin University of China &nbsp;
    <sup>2</sup>AMap, Alibaba &nbsp;
    <sup>3</sup>Tsinghua University &nbsp;
    <sup>4</sup>Wuhan University &nbsp;
    <sup>5</sup>Malou Tech Inc
    <br><br>
    <sup>*</sup>Project Leader &nbsp;
    <sup>†</sup>Corresponding Authors
</p>

  <p align="center">
    <a href="https://arxiv.org/abs/2512.18181">
      <img src="https://img.shields.io/badge/arXiv-MACE--Dance-green" alt="Paper">
    </a>
    <a href="https://sun-happy-ykx.github.io/MACE-Dance/">
      <img src="https://img.shields.io/badge/Project_Page-MACE--Dance-blue" alt="Project Page">
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/Conference-SIGGRAPH%202026-orange" alt="Conference">
    </a>
  </p>
</p>

<p align="center">
  <img src="teaser.png" width="90%" alt="MACE-Dance teaser">
</p>

<p align="center">
  <em>
    MACE-Dance is a cascaded expert framework for music-driven dance video generation,
    explicitly decoupling motion generation and appearance synthesis to produce
    kinematically plausible, artistically expressive, and visually coherent dance videos.
  </em>
</p>

---

## ✨ Overview

**MACE-Dance** is the official PyTorch implementation of the SIGGRAPH 2026 paper:

> **MACE-Dance: Motion-Appearance Cascaded Experts for Music-Driven Dance Video Generation**

Music-driven dance video generation is challenging because it requires simultaneously modeling:

- **Motion quality**: kinematically plausible and artistically expressive dance motion
- **Appearance quality**: high-fidelity visual synthesis with strong spatiotemporal consistency

To address this, **MACE-Dance** decomposes the task into two cascaded experts:

- **🕺 Motion Expert**: generates music-aligned **3D dance motion**
- **🎨 Appearance Expert**: synthesizes the final **dance video** conditioned on motion and reference appearance

Instead of using 2D keypoints as the intermediate representation, MACE-Dance adopts **3D SMPL motion**, which provides better spatial fidelity, cleaner supervision, and stronger robustness for downstream video synthesis.

---

## 🧩 Repository Structure

```bash
MACE-Dance/
├── Expert-Motion/           # Motion Expert: music-to-3D dance motion
├── Expert-Appearance/       # Appearance Expert: motion-guided video synthesis
├── Evaluation-Motion/       # Motion-dimension evaluation
├── Evaluation-Appearance/   # Appearance-dimension evaluation
├── teaser.png
└── README.md
```
---

## 📚 MA-Data Dataset

We provide **MA-Data**, a large-scale dataset for music-driven dance video generation, containing **~70K video clips** spanning **116 hours** across **20+ dance genres**. Please refer to the [dataset page](https://huggingface.co/datasets/GD-ML/MACE-Dance) for more details.


---

## 🏋️ Model Weights

The source code for **MACE-Dance** is fully open-source. For the model weights of the Appearance Expert, please visit the link below to request access or download:

👉 **[Click here to access MACE-Dance Model Weights](https://huggingface.co/GD-ML/MACE-Dance)**

---

## 📏 Evaluation Protocol

We provide a motion–appearance evaluation protocol for music-driven dance video generation, including motion quality assessment based on ViTPose keypoints and appearance quality assessment based on VBench. Please refer to `Evaluation-Motion` and `Evaluation-Appearance` for details.

---

## 📌 Notes

- The repository is organized into **expert modules** and **evaluation modules**.
- Please check the subfolder READMEs for environment setup, inference, and evaluation details.
- Some released example files are for demonstration only; please replace them with your own predictions / ground-truth files during evaluation.

---

## 🛠️ Local Setup Notes (tested on a single RTX 4090, 24 GB)

These notes are specific to this fork and are not from the original authors.

**Weights.** The MACE weights are not gated, but are packaged as `.7z` archives in
<https://huggingface.co/GD-ML/MACE-Dance> (`Expert-Appearance/Expert-Appearance.7z`,
`Expert-Motion/Expert-Motion.7z`). Extract them with `py7zr` (or `7z`):

- Appearance: the archive contains `model/train/{Adapter,LoRA}.safetensors`; place them in
  `Expert-Appearance/models/train/`. The Wan2.2-Animate-14B base model (~42 GB of the
  repo's 72 GB is needed) goes in `Expert-Appearance/models/Wan-AI/Wan2.2-Animate-14B/`.
- Motion: extract into `Expert-Motion/`. It provides `runs/train/exp/weights/train-3750.pt`,
  the `data/dataset_backups/` cache and test wavs under `data/test/wavs/`.

**Environments.** The two experts need separate environments (Appearance: recent torch;
Motion: torch 2.4.0). In both, `transformers` must be `<5`; `transformers` 5.x removed APIs
that `diffsynth` and `mamba-ssm` import. For the Motion Expert, install the prebuilt wheels
rather than compiling (CUDA 12, torch 2.4, Python 3.10):

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
            "https://github.com/state-spaces/mamba/releases/download/v2.2.6.post2/mamba_ssm-2.2.6.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install pytorch3d==0.7.8 --find-links https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html
```

**GPU memory.** The Appearance Expert originally loaded the full 14B model onto the GPU,
which runs out of memory on 24 GB. `appearance_infer.py` now loads models on the CPU and
relies on `enable_vram_management()` to stream layers to the GPU. On a 4090 this used
~24 GB of VRAM and ~43 GB of system RAM, and took ~20 minutes for one 1280×720, 77-frame
clip. Larger GPUs should be faster; this was not measured.

**Scope of the Motion Expert.** `Expert-Motion/test.py` evaluates on the released test set
and writes SMPL motion to `eval/`. It does not take an arbitrary song, and it does not render
video. The Appearance Expert's pose/face control videos are provided only as samples in
`Expert-Appearance/data/`.

---

## 📄 Citation

If you find this project useful, please consider citing our paper:

```bibtex
@inproceedings{yang2026macedance,
  title={MACE-Dance: Motion-Appearance Cascaded Experts for Music-Driven Dance Video Generation},
  author={Yang, Kaixing and Zhu, Jiashu and Tang, Xulong and Peng, Ziqiao and Zhang, Xiangyue and Wang, Puwei and Wu, Jiahong and Chu, Xiangxiang and Liu, Hongyan and He, Jun},
  booktitle={Proceedings of the ACM SIGGRAPH Conference},
  year={2026}
}
```

---

## 🙏 Acknowledgement

This work was supported in part by the National Nature Science Foundation of China under Grants **62436010**, **72572090**, **62572474**, and **62172421**, and in part by the Tsinghua University School of Economics and Management Research Grant.
