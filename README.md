# Dailies Scene AI

Research-style, modular Python project to organize film dailies (shots/takes) into scene groups via a multi-stage AI pipeline.

## Setup (Mac M1/M2)

Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

The canonical entrypoint is:

```bash
python run.py
```

This will execute the pipeline stages in the same order as the final system:

1. Shot preprocessing (frame extraction)
2. Face detection
3. Actor recognition (planned)
4. Scene context encoding (planned)
5. Temporal shot encoder (planned)
6. Scene clustering (planned)

`main.py` is kept as a thin wrapper around `run.py` for backwards compatibility.

## Stage 1 (implemented): Frame extraction

1. Put your raw shots into `data/raw_shots/` (e.g., `take_001.mov`, `take_002.mov`).
2. Run:

```bash
python run.py --config configs/config.yaml --raw-shots data/raw_shots --frames-out data/frames
```

Outputs are written under `data/frames/<shot_id>/`:

- `sampled/`: uniformly sampled frames at `frame_sampling_rate_fps`
- `keyframes/`: a subset based on lightweight scene-change scoring (optional)
- `manifest.json`: metadata + file lists

## Pipeline stages

The repo is scaffolded for the full pipeline:

1. Shot preprocessing (frame sampling + keyframes) ✅
2. Face detection (MTCNN) ✅
3. Actor recognition (ArcFace-style embeddings) ⏳
4. Scene context encoding (CLIP ViT) ⏳
5. Temporal shot encoder (trainable Transformer) ⏳
6. Scene clustering (HDBSCAN) ⏳

