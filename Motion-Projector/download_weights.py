"""Download the ViTPose-H wholebody and YOLOv10m ONNX weights used by the projector.

They live in the Wan2.2-Animate-14B repo under process_checkpoint/. The ViTPose "file" is
really a folder (~390 files, 2.5 GB) that `hf download --include` does not match, so each
file is fetched individually. Requires `huggingface_hub`.
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Wan-AI/Wan2.2-Animate-14B"
LOCAL_DIR = Path(__file__).resolve().parent / "models"
VITPOSE_DIR = "process_checkpoint/pose2d/vitpose_h_wholebody.onnx"
DETECTOR = "process_checkpoint/det/yolov10m.onnx"


def main():
    url = f"https://huggingface.co/api/models/{REPO}/tree/main/{VITPOSE_DIR}"
    files = [f["path"] for f in json.load(urllib.request.urlopen(url)) if f["type"] == "file"]
    files.append(DETECTOR)
    print(f"Downloading {len(files)} files to {LOCAL_DIR}", flush=True)

    def fetch(path):
        hf_hub_download(REPO, path, local_dir=str(LOCAL_DIR))

    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(fetch, files), 1):
            if i % 50 == 0:
                print(f"{i}/{len(files)}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
