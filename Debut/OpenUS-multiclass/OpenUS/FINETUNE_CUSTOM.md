# Fine-tuning OpenUS segmentation on the custom echography dataset

Procedure to fine-tune the OpenUS-S (VMamba-small) segmentation pipeline on the custom COCO dataset
in `data/` (169 images, 3 overlapping classes: Cavite, Gonade, Intestin — **multi-label**: one pixel can
belong to several classes). Only the MambaDecoder head trains; the OpenUS backbone stays frozen.

## What you need to copy to the training machine

- This repository **including the local modifications** (`dataset/dataset_custom_coco.py`,
  `dataset/transforms.py`, `eval_segmentation.py`, `test_segmentation.py`) — commit or rsync the tree.
- `data/` — `splits.json` (generated from the shared dataset and common test fish, see step 2).
- `checkpoint/openus_cpt0150.pth` — the pre-trained OpenUS-S weights.

## 1. Environment (Debian/Ubuntu server or any standard Linux + NVIDIA GPU)

### 1a. System prerequisites (once, needs sudo)

```bash
# Check what's already there
nvidia-smi                 # driver working? (any driver supporting CUDA 12.x, i.e. >= 525)
nvcc --version             # CUDA toolkit present? (needed to BUILD the fast kernel, 12.x)
gcc --version              # host compiler for the kernel build

# If missing, on Debian/Ubuntu:
sudo apt update
sudo apt install -y build-essential git tmux
# CUDA toolkit — easiest via NVIDIA's apt repo (https://developer.nvidia.com/cuda-downloads)
# e.g. for Ubuntu 22.04:
#   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
#   sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
#   sudo apt install -y cuda-toolkit-12-4
#   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

Notes:
- The *driver* (what `nvidia-smi` reports) and the *toolkit* (`nvcc`) are separate installs; a server
  often has only the driver. You need the toolkit only to compile `selective_scan` — training itself
  runs on pip-installed CUDA libraries.
- The toolkit's major.minor doesn't need to match the driver exactly; any CUDA 12.x toolkit is fine
  with torch 2.2's cu121 builds.
- Headless is fine — nothing here needs a display.

### 1b. Python environment

With conda (README's recipe):

```bash
conda create -n openus python=3.10 -y
conda activate openus
pip install torch==2.2 torchvision torchaudio triton pytest chardet yacs termcolor fvcore seaborn packaging ninja einops numpy==1.24.4 timm==0.4.12
pip install pyiqa clean-fid pandas openpyxl scikit-image
```

No conda on the server? Plain venv works the same (needs `sudo apt install python3.10-venv` on
Ubuntu 22.04, or `pyenv`/deadsnakes if the system python isn't 3.10 — the repo is pinned to 3.10):

```bash
python3.10 -m venv ~/openus-venv
source ~/openus-venv/bin/activate
pip install --upgrade pip
pip install torch==2.2 torchvision torchaudio triton pytest chardet yacs termcolor fvcore seaborn packaging ninja einops numpy==1.24.4 timm==0.4.12
pip install pyiqa clean-fid pandas openpyxl scikit-image
```

Sanity check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

### 1c. Fast selective-scan kernel (strongly recommended)

Build the fast selective-scan CUDA kernel (this is what needs `nvcc` from step 1a). Without it,
VMamba falls back to a pure-PyTorch scan that is ~10-50× slower and far more memory-hungry
(on a 6 GB GTX 1660 Ti only batch 2 fits and one epoch takes ~40 min):

```bash
cd vmamba_models/selective_scan
pip install .          #  sudo apt install python3.10-dev and python -m pip install .
cd ../..
python -c "import selective_scan_cuda_oflex; print('fast kernel OK')" #python -c "import torch; import selective_scan_cuda_core; print('Kernel Core OK')"
```

Do **not** rely on the prebuilt `mamba_ssm` wheel from the README for this: its compiled extension has
an ABI mismatch with pip's torch 2.2 (`undefined symbol` on import) — building `selective_scan` from
source is the reliable route.

## 2. Synchronize data and the global test split

From this directory, run the standalone command (Python standard library only):

```bash
python sync_custom_data.py --dry-run  # validate sources without writing
python sync_custom_data.py
```

The command reads `Debut/data/COCO/images/cavite` and
`Debut/data/COCO/labels/cavite/annotations.json` directly. It validates the sources and
writes only the split files into local `data/`; no images or annotations are copied.
Use `--debut-root /path/to/Debut` if the shared inputs are elsewhere.

Test images are exactly those belonging to the fish listed in
`Debut/utils/common_test_fish.json` (third underscore-separated field of each image
name). The remaining fish IDs are sorted, shuffled with seed 42, and the first
`round(20% * number_of_remaining_fish)` are reserved for validation. No fish is
shared between splits. Current counts: **118 train / 27 val / 24 test**.
`data/splits_smoke.json` contains the first sorted filename from each split.

The three channels remain Cavite, Gonade, Intestin. Category `ignore` polygons
are skipped by the existing loader; their pixels are not excluded from losses
or metrics. Checkpoints and previous results are retained and describe the old
87-image dataset: use a new output directory, such as `output/custom_seg_new`,
for training and subsequent testing on the updated dataset.

Do not use the older `dataset.dataset_custom_coco --make_splits` command for this
workflow: it splits by image and does not honor the common test fish.

## 3. Training

On a remote server, run inside `tmux` (or `nohup ... &`) so an SSH disconnect doesn't kill the job,
and copy the inputs first:

```bash
# From this machine:
rsync -av --exclude .venv --exclude output /home/victorj/Travail/OpenUS/ user@server:~/OpenUS/

# On the server:
tmux new -s openus
conda activate openus     # or: source ~/openus-venv/bin/activate
cd ~/OpenUS
```

```bash
python eval_segmentation.py \
  --arch vmamba_small --dataset_name CUSTOM --multilabel True \
  --coco_json ../../data/COCO/labels/cavite/annotations.json \
  --images_root ../../data/COCO/images/cavite \
  --split_file data/splits.json \
  --pretrained_vmamba True --pretrained_weights checkpoint/openus_cpt0150.pth \
  --checkpoint_key teacher \
  --epochs 100 --lr 0.001 --batch_size_per_gpu 8 --num_workers 4 \
  --output_dir output/custom_seg --log_name custom_seg
```

Flags that must not change (they look optional but aren't):
- `--pretrained_vmamba True` — the only backbone construction that returns the multi-scale features
  the MambaDecoder head needs; `False` crashes. The `./pretrained/vmamba/vssm_*.pth` ImageNet file it
  mentions is *optional* (a warning is printed if absent; the OpenUS checkpoint overwrites the backbone).
- `--checkpoint_key teacher` — the `student` weights in the checkpoint have a different key prefix and
  would silently not load.
- `--num_classes` is inferred from the COCO categories. To train Cavite/Gonade only,
  use `annotations_2_classes.json` and a separate output directory; an explicit
  `--num_classes` remains accepted and is checked against the JSON.
- `--multilabel True` — switches to BCE loss + per-class sigmoid metrics; without it the CrossEntropy
  path mislabels the multi-channel masks.

Input resolution: CUSTOM defaults to **512×512** (other datasets stay at 224×224); override with
`--img_size` in both `eval_segmentation.py` and `test_segmentation.py`. Saved masks and overlays
come out at that same resolution. 512² is ~5× the pixels of 224², so expect proportionally more
VRAM and slower epochs.

If you hit `CUDA out of memory`, lower `--batch_size_per_gpu` (16 → 8 → 4 → 2) and optionally
`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Multi-GPU is unnecessary here (118 training images = 8 steps/epoch at batch 16); a single GPU is the
right choice. Detach from tmux with `Ctrl-b d`, reattach later with `tmux attach -t openus`, and watch
progress with `tail -f output/custom_seg/log_custom_seg.txt`.

Outputs in `output/custom_seg/`:
- `checkpoint_teacher_seg_best.pth` — best head weights by validation mIoU/Dice (this is the one to keep)
- `checkpoint_teacher_seg_epoch_N.pth` — periodic snapshots every 10 epochs
- `log_custom_seg.txt` — one JSON line per epoch: train/val loss, mean + per-class IoU and Dice
  (per-class order: [Cavite, Gonade, Intestin])

## 4. Held-out test evaluation

Run after training, from the same `--output_dir` (it picks up `checkpoint_teacher_seg_best.pth`):

```bash
python test_segmentation.py \
  --arch vmamba_small --dataset_name CUSTOM --multilabel True \
  --coco_json ../../data/COCO/labels/cavite/annotations.json \
  --images_root ../../data/COCO/images/cavite \
  --split_file data/splits.json \
  --pretrained_vmamba True --pretrained_weights checkpoint/openus_cpt0150.pth \
  --checkpoint_key teacher \
  --batch_size_per_gpu 4 \
  --output_dir output/custom_seg
```

Outputs: `test_results_teacher_best.json` (mean + per-class IoU/Dice over the 24 test images, plus a
`per_image` list with IoU, Dice, precision, recall and surface difference for every test image),
`test_results_per_image_teacher_best.csv` (named per-class columns such as `dice_gonade`),
`predicted_masks_teacher/` (RGB pngs, R=Cavite, G=Gonade, B=Intestin — same encoding as the training
masks, so overlapping classes show as mixed colors) and `overlay_masks_teacher/` (test images with
the predicted masks blended on top; opacity via `--overlay_alpha`, default 0.4).

## NixOS (this machine) specifics

Everything above works via the flake, with two extra env vars and a small batch:

```bash
nix develop
uv sync --extra vmamba
TRITON_LIBCUDA_PATH=/run/opengl-driver/lib \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python eval_segmentation.py ... --batch_size_per_gpu 2 ...
```

`TRITON_LIBCUDA_PATH` is required because triton's kernel builder shells out to `/sbin/ldconfig`,
which doesn't exist on NixOS. Batch 2 is the largest that fits in 6 GB with the pure-PyTorch scan
(~40 min/epoch on the GTX 1660 Ti — usable for smoke tests, not for the full 100-epoch run).
The fast-kernel build (step 1) is not available here without adding `cudatoolkit` + `gcc` to `flake.nix`.

## Implementation notes (what was added to the repo)

- `dataset/dataset_custom_coco.py` — `CocoMultiLabelDataset`: rasterizes COCO polygons per class into
  the R/G/B channels of one mask image (multi-label safe); plus the `--make_splits` CLI.
- `dataset/transforms.py` — `get_transforms_multilabel()`: same aug pipeline, mask kept as
  `[3,H,W]` float {0,1} instead of collapsed to integer labels.
- `eval_segmentation.py` / `test_segmentation.py` — `CUSTOM` dataset branch, `--multilabel` mode
  (BCEWithLogitsLoss, sigmoid>0.5 predictions, per-class + mean Dice/IoU), RGB prediction saving.
  The existing TN3K/BUSBRA paths are untouched.
