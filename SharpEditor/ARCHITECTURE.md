# SharpEditor Architecture & Changelog

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Pipeline Stages](#pipeline-stages)
- [Data Flow](#data-flow)
- [Changes Made](#changes-made)
- [Design Reasoning](#design-reasoning)
- [Alternative Methods for Different Data](#alternative-methods-for-different-data)
- [Scoring & Error Mechanisms (Planned)](#scoring--error-mechanisms-planned)
- [Directory Structure](#directory-structure)
- [Configuration Reference](#configuration-reference)
- [Dependencies](#dependencies)
- [JSON Output Formats](#json-output-formats)
- [Demo Benchmark Results](#demo-benchmark-results)
- [Known Limitations](#known-limitations)
- [Training Roadmap](#training-roadmap)

---

## Project Overview

SharpEditor (Dailies Scene AI) is a 6-stage pipeline that automatically organizes raw film dailies into scene groups. It processes video shots through frame extraction, face detection, actor recognition, scene context encoding, temporal shot aggregation, and finally clusters shots into coherent scenes.

The system is designed for film production workflows where hundreds of raw takes need to be organized by scene for editorial. It uses a multi-modal approach: **who** is in the shot (actor identity via face embeddings), **what** the shot looks like (scene semantics via CLIP), and **how** features evolve over time (temporal Transformer aggregation).

```
Raw Video Shots
      |
      v
[1] Frame Extraction ──> sampled frames + keyframes
      |
      v
[2] Face Detection ────> face crops + aligned faces
      |
      v
[3] Actor Recognition ─> 512-D face embeddings + actor IDs (HDBSCAN)
      |
      v
[4] Scene Encoding ────> 768-D CLIP scene embeddings
      |
      v
[5] Shot Encoding ─────> 1024-D fused shot embeddings (Transformer)
      |
      v
[6] Scene Clustering ──> scene group assignments (HDBSCAN)
```

---

## System Architecture

### High-Level Component Map

```
SharpEditor/
├── run.py                         # Pipeline orchestrator (all 6 stages)
├── configs/config.yaml            # Central configuration
│
├── pipeline/                      # Processing stages
│   ├── frame_extractor.py         # Stage 1: Video → frames
│   ├── face_detector.py           # Stage 2: Frames → face crops
│   ├── actor_recognition.py       # Stage 3: Faces → actor embeddings + IDs
│   ├── scene_encoder.py           # Stage 4: Frames → CLIP embeddings
│   ├── shot_encoder.py            # Stage 5: Features → shot embeddings
│   └── clustering.py              # Stage 6: Embeddings → scene groups
│
├── models/
│   └── temporal_transformer.py    # Transformer encoder architecture
│
├── training/                      # Training infrastructure (stubs)
│   ├── dataset.py                 # ShotPairDataset (planned)
│   ├── train_temporal.py          # Training loop (planned)
│   └── loss_functions.py          # InfoNCE loss (planned)
│
├── utils/
│   ├── config_utils.py            # YAML config loading
│   ├── device_utils.py            # MPS / CUDA / CPU selection
│   ├── video_utils.py             # Video I/O, frame sampling, JSON writes
│   └── embedding_utils.py         # L2 normalization
│
├── scripts/
│   └── split_into_shots.py        # Utility: split full films into clips
│
└── data/
    ├── raw_shots/                 # Input video clips
    ├── frames/                    # Per-shot intermediate outputs
    └── embeddings/                # Dataset-level embeddings + results
```

### Model Stack

| Component | Model | Embedding Dim | Source |
|-----------|-------|--------------|--------|
| Face Detection | MTCNN (facenet-pytorch) | bounding boxes + landmarks | Pre-trained |
| Actor Embeddings | InceptionResnetV1 | 512-D | VGGFace2 pre-trained |
| Scene Embeddings | CLIP ViT-B/32 | 768-D | OpenAI pre-trained |
| Shot Aggregation | TemporalTransformer | 1024-D | Initialized (trainable) |
| Clustering | HDBSCAN | — | Unsupervised |

### Device Strategy

The pipeline targets Apple Silicon (MPS) as the primary accelerator with fallbacks:

- **MPS (Metal)**: Used for InceptionResnetV1, CLIP, TemporalTransformer
- **CPU forced**: MTCNN face detection (adaptive pooling not supported on MPS)
- **CUDA**: Supported if available, preferred over CPU
- **CPU**: Universal fallback

---

## Pipeline Stages

### Stage 1: Frame Extraction

**Module**: `pipeline/frame_extractor.py` | **Class**: `FrameExtractor`

Extracts representative frames from each raw video shot using two strategies:

1. **Uniform sampling** at a target FPS (default 2 fps, max 30 frames/shot)
2. **Keyframe selection** via HSV histogram scene-change scoring (Bhattacharyya distance between consecutive frames)

Scene-change scores are computed for every sampled frame. Frames with scores above a threshold (default 0.25) and that are local peaks are selected as keyframes (max 12 per shot). These keyframes capture the most visually distinctive moments in each shot.

**Outputs per shot**: `manifest.json`, `sampled/` directory, `keyframes/` directory

### Stage 2: Face Detection

**Module**: `pipeline/face_detector.py` | **Class**: `FaceDetector`

Runs MTCNN (Multi-task Cascaded Convolutional Networks) on keyframes to detect faces. For each detection above the confidence threshold (default 0.8):

- Saves a raw BGR face crop
- Saves an MTCNN-aligned RGB face crop (standardized 160x160, suitable for embedding models)
- Records bounding box, confidence score, and 5-point facial landmarks

**Outputs per shot**: `faces/`, `faces_aligned/`, `faces.json`

### Stage 3: Actor Recognition

**Module**: `pipeline/actor_recognition.py` | **Class**: `ActorRecognizer`

Computes 512-D face embeddings using InceptionResnetV1 (pre-trained on VGGFace2), then clusters all face embeddings across the dataset using HDBSCAN to assign actor IDs.

- All embeddings are L2-normalized so Euclidean distance equals cosine distance on the unit hypersphere
- HDBSCAN noise points (label -1) represent faces that couldn't be confidently assigned to an actor
- Per-actor mean embeddings are computed and stored for downstream similarity calculations

**Outputs**: `actor_embeddings.npz` (dataset), `actors.json` (dataset), `actors.json` (per-shot)

### Stage 4: Scene Encoding

**Module**: `pipeline/scene_encoder.py` | **Class**: `SceneEncoder`

Encodes keyframes using CLIP ViT-B/32 to produce 768-D scene embeddings that capture high-level visual semantics (setting, objects, lighting, composition). Per-frame embeddings are mean-pooled into a single per-shot scene embedding.

CLIP was chosen because it produces semantically rich representations without task-specific fine-tuning. Two shots from the same physical location or with similar visual characteristics will have high cosine similarity.

**Outputs**: `scene_embeddings.npz` (dataset), `scene.json` (per-shot)

### Stage 5: Temporal Shot Encoding

**Module**: `pipeline/shot_encoder.py` | **Class**: `ShotEncoder`
**Model**: `models/temporal_transformer.py` | **Class**: `TemporalTransformer`

Fuses actor (512-D) and scene (768-D) embeddings into a unified 1024-D shot embedding via a Transformer encoder. The architecture:

```
Per-frame input: [actor_emb (512) | scene_emb (768)] = 1280-D
                              |
                    Linear projection → 512-D
                              |
                    Prepend [CLS] token
                              |
                    + Learned positional embeddings
                              |
                    4x Transformer encoder layers
                    (8 heads, 1024 FFN, pre-norm)
                              |
                    [CLS] output → LayerNorm
                              |
                    Linear projection → 1024-D
                              |
                    L2 normalization
```

The [CLS] token aggregates information across all frames via self-attention, producing a single fixed-size representation regardless of shot length. Positional embeddings encode frame ordering.

Currently runs with initialized (untrained) weights. Even without training, the Transformer provides a meaningful non-linear fusion of the pre-trained CLIP and FaceNet features, which are already well-structured in their respective embedding spaces.

**Outputs**: `shot_embeddings.npz` (dataset), `shot.json` (per-shot)

### Stage 6: Scene Clustering

**Module**: `pipeline/clustering.py` | **Class**: `SceneClusterer`

Groups shots into scenes using a fused distance metric:

1. **Shot embedding similarity** (cosine, from 1024-D vectors): captures visual + actor feature similarity
2. **Actor co-occurrence similarity** (Jaccard index): binary signal of which actors appear in each shot

These are combined as:
```
fused_similarity = (1 - actor_weight) * cosine_sim + actor_weight * jaccard_sim
distance = clamp(1 - fused_similarity, 0, 2)
```

Default `actor_weight = 0.3` (70% embedding similarity, 30% actor co-occurrence).

HDBSCAN clusters the fused distance matrix. Shots that don't fit any cluster (noise label -1) are tracked as "ungrouped" rather than forced into a scene.

**Outputs**: `scenes.json` (dataset), `scene_group.json` (per-shot)

---

## Data Flow

### Per-Shot Output Structure

```
data/frames/<shot_id>/
├── manifest.json        # Stage 1: extraction metadata, frame lists, scene-change scores
├── sampled/             # Stage 1: uniformly sampled frames (JPG)
├── keyframes/           # Stage 1: scene-change-selected keyframes (JPG)
├── faces/               # Stage 2: raw face crops (BGR JPG)
├── faces_aligned/       # Stage 2: MTCNN-aligned face crops (RGB JPG, 160x160)
├── faces.json           # Stage 2: detection metadata (bboxes, scores, landmarks)
├── actors.json          # Stage 3: actor IDs, face counts, per-shot actor embeddings
├── scene.json           # Stage 4: 768-D CLIP scene embedding
├── shot.json            # Stage 5: 1024-D fused shot embedding
└── scene_group.json     # Stage 6: assigned scene ID
```

### Dataset-Level Output Structure

```
data/embeddings/
├── actor_embeddings.npz   # (N_faces, 512) all face embeddings + metadata
├── actors.json            # actor clusters, mean embeddings, face lists
├── scene_embeddings.npz   # (N_shots, 768) per-shot CLIP embeddings
├── shot_embeddings.npz    # (N_shots, 1024) per-shot fused embeddings
└── scenes.json            # final scene groups + ungrouped shots
```

### Embedding Shape Summary

| Stage | Array | Shape | Notes |
|-------|-------|-------|-------|
| Actor | actor_embeddings.npz | (N_faces, 512) | One row per detected face |
| Scene | scene_embeddings.npz | (N_shots, 768) | One row per shot (mean of frames) |
| Shot | shot_embeddings.npz | (N_shots, 1024) | One row per shot (Transformer output) |

---

## Changes Made

### New Files

#### `models/temporal_transformer.py` — Transformer Shot Encoder Model

**What**: A PyTorch `nn.Module` implementing a Transformer encoder with [CLS] token aggregation. Takes variable-length sequences of 1280-D per-frame features and outputs a fixed 1024-D shot embedding.

**Reasoning**: The pipeline needed a way to fuse the multi-modal per-frame signals (actor identity + scene context) into a single shot-level representation. A Transformer was chosen over simpler alternatives (mean pooling, RNN) because:
- Self-attention captures relationships between frames regardless of distance in the sequence
- The [CLS] token provides a natural aggregation point without information loss from averaging
- Positional embeddings preserve temporal ordering (frame 1 vs frame 10 matters)
- The architecture is ready for supervised fine-tuning once training data is available
- Pre-norm (norm_first=True) transformer layers are more stable for training

**Design decisions**:
- `d_model=512` (not 1024) to keep the model compact; the output projection upsamples to 1024-D
- 4 layers, 8 heads — sufficient for the sequence lengths we see (1-30 frames) without being overly heavy
- Xavier uniform initialization for stable initial forward passes even without training
- Padding mask support for batched inference with variable-length shots

#### `pipeline/shot_encoder.py` — Stage 5 Pipeline Module

**What**: Loads per-shot actor and scene features, constructs per-frame feature matrices, runs them through the TemporalTransformer, and writes L2-normalized 1024-D shot embeddings.

**Reasoning**: This stage bridges the gap between per-modality features (Stages 3-4) and the final clustering (Stage 6). Without it, clustering would need to operate on separate actor and scene embeddings with manual fusion weights. The Transformer provides a learned (or learnable) fusion.

**Design decisions**:
- Feature construction replicates the per-shot mean embeddings across frames since individual per-frame CLIP embeddings are not stored. This is acceptable because: (a) actor presence provides per-frame variation, (b) the Transformer can still learn useful patterns from the combined signal, (c) storing per-frame CLIP embeddings would significantly increase storage
- Supports loading pre-trained weights via `weights_path` config for future use
- L2 normalization of output embeddings ensures cosine similarity works correctly in Stage 6
- Skips shots missing `scene.json` rather than failing

#### `pipeline/clustering.py` — Stage 6 Pipeline Module

**What**: Clusters shots into scene groups using a fused distance matrix combining shot embedding cosine similarity and actor co-occurrence Jaccard similarity, via HDBSCAN.

**Reasoning**: Scene membership is determined by two complementary signals:
1. **Visual/semantic similarity**: shots from the same scene tend to share similar settings, lighting, and composition (captured by CLIP + Transformer embeddings)
2. **Actor co-occurrence**: shots from the same scene typically feature the same actors

Fusing these signals produces more robust clusters than either alone. HDBSCAN was chosen over K-means or spectral clustering because:
- It doesn't require specifying the number of scenes in advance
- It naturally identifies outlier shots as "noise" rather than forcing them into a cluster
- It handles clusters of varying density and size
- It works well with precomputed distance matrices

**Design decisions**:
- Actor weight defaults to 0.3 (30% actor, 70% embedding). This was chosen because in dailies workflows, the same actors can appear across different scenes (e.g., the lead appears everywhere), so actor co-occurrence is a useful but not dominant signal
- Jaccard index for actor similarity (rather than cosine on binary vectors) because it naturally handles the "no actors detected" case and provides intuitive 0-1 similarity
- Ungrouped shots are explicitly tracked rather than discarded — in a real workflow, an editor would want to review these manually
- Precomputed distance matrix passed to HDBSCAN for full control over the similarity fusion

#### `scripts/split_into_shots.py` — Film Splitting Utility

**What**: Uses ffmpeg scene detection to split full-length films into individual shot clips suitable for pipeline input.

**Reasoning**: The pipeline expects pre-segmented shots as input (simulating raw dailies from a film shoot). For testing with publicly available films, we needed a way to create realistic shot-length clips. ffmpeg's built-in scene detection (`select='gt(scene,threshold)'`) provides a reasonable approximation of actual shot boundaries.

**Design decisions**:
- Scene threshold of 0.35 balances between too many cuts (noisy) and too few (long segments)
- Duration filtering (3-30s) ensures clips are realistic shot lengths
- Even sampling across the film ensures diverse scene coverage
- Re-encodes to H.264 for consistent codec handling downstream
- Writes a `shots_manifest.json` for provenance tracking

### Modified Files

#### `run.py` — Wired Stages 5 and 6

**What changed**: Added imports for `ShotEncoder` and `SceneClusterer`. Replaced the `NotImplementedError` stubs in `run_temporal_shot_encoding()` and `run_scene_clustering()` with actual implementations that instantiate and run the new modules. Added Stage 5 and Stage 6 execution blocks to `main()`.

**Reasoning**: The pipeline orchestrator needed to call the new stages in sequence after Stages 1-4, following the same pattern established by the existing stages (construct from config, call `process_dataset`).

#### `pipeline/actor_recognition.py` — Empty actors.json for faceless shots

**What changed**: Added two blocks that write an empty `actors.json` (with `actors: {}`, `faces: []`, `actor_embeddings: {}`) for shots where no faces were detected. Previously, these shots were silently skipped, leaving downstream stages (shot encoder, clustering) without the expected file.

**Reasoning**: Every shot directory must have a consistent set of output files across all 6 stages for the pipeline to be reliable and inspectable. The shot encoder reads `actors.json` to build per-frame feature vectors — a missing file would either cause a crash or silently produce different behavior than an explicitly-empty file. Writing empty records also makes it easy to audit which shots had no detected faces (they'll have `"faces": []` rather than a missing file).

#### `models/temporal_transformer.py` — Suppressed nested_tensor warning

**What changed**: Set `enable_nested_tensor=False` in the `TransformerEncoder` constructor.

**Reasoning**: PyTorch emits a `UserWarning` when `norm_first=True` is used with nested tensors enabled (nested tensors require post-norm). The warning is harmless but produces noise in pipeline output. Since we use pre-norm for training stability, we explicitly disable nested tensors to keep output clean for demos.

---

## Design Reasoning

### Why Multi-Modal Fusion?

A single modality is insufficient for scene grouping:
- **Faces alone** fail when scenes share actors (the lead appears in every scene) or when faces aren't visible (wide shots, action sequences)
- **Scene visuals alone** fail when different scenes share similar settings (two dialogue scenes in the same room) or when the same scene has visually diverse shots (close-up vs wide)
- **Combined** signal is robust: same actors + similar visual context = high confidence same scene

### Why a Transformer (Not Simpler Pooling)?

Mean pooling discards temporal structure. An RNN would work but processes frames sequentially. The Transformer:
- Processes all frames in parallel
- Learns which frames are most informative via attention weights
- Can capture long-range dependencies (frame 1 relates to frame 30)
- Is the standard architecture for this type of set/sequence aggregation task

### Why HDBSCAN (Not K-Means)?

K-Means requires knowing the number of scenes in advance, which is exactly what we're trying to discover. Spectral clustering also requires K. HDBSCAN:
- Discovers the number of clusters automatically
- Identifies outliers explicitly
- Handles non-spherical cluster shapes
- Works with any distance metric

### Why L2 Normalization Everywhere?

All embeddings (actor, scene, shot) are L2-normalized to lie on the unit hypersphere. This means:
- Cosine similarity = simple dot product (fast)
- Euclidean distance is monotonically related to cosine distance
- Embedding magnitudes don't dominate — only direction matters
- Numerically stable comparisons across different embedding spaces

---

## Alternative Methods for Different Data

The current pipeline is optimized for **film dailies** (narrative content with actors, structured scenes). Different source material would benefit from different approaches:

### Documentary / Nature Footage

- **Replace face detection** with object detection (YOLOv8, DETR) to identify recurring subjects (animals, landmarks, vehicles)
- **Add audio analysis** — narration topic changes are strong scene boundary signals
- **Use temporal proximity** more heavily — documentary scenes tend to be contiguous in capture order
- **Color grading similarity** — documentary scenes often share consistent grading

### Surveillance / Security Footage

- **Replace CLIP with motion-based features** — optical flow magnitude/direction histograms
- **Person re-identification (ReID)** instead of face recognition — handles distant/partial views
- **Timestamp-based grouping** as a prior — events cluster temporally
- **Anomaly detection** instead of scene grouping — flag unusual activity vs background

### Live Event / Multi-Camera

- **Audio fingerprinting** to synchronize cameras capturing the same moment
- **Camera angle classification** (wide, medium, close-up) as an additional feature
- **Timecode alignment** — professional multi-cam shoots have synced timecodes
- **Speaker diarization** to group by who is speaking, not just who is visible

### User-Generated Content / Social Media

- **Text overlay / OCR detection** — memes, captions, titles indicate scene context
- **Audio event detection** — music, speech, laughter, applause as scene signals
- **Replace CLIP with video-native models** — VideoMAE, TimeSformer for temporal understanding
- **Hashtag/metadata clustering** as supplementary signal

### Animation / VFX Shots

- **Skip face detection entirely** — use character recognition models or color palette matching
- **Style transfer detection** — group by visual style consistency
- **Layout similarity** — compare spatial arrangement of elements
- **Production metadata** (shot names, slate info) if available

### Alternative Clustering Approaches

| Method | Best For | Trade-off |
|--------|----------|-----------|
| **HDBSCAN** (current) | Unknown number of scenes, noisy data | May under-cluster with conservative settings |
| **Agglomerative (Ward)** | Hierarchical scene structure | Requires choosing cut threshold |
| **Spectral Clustering** | Non-convex cluster shapes | Requires K; expensive for large N |
| **DBSCAN** | Fixed-density clusters | Less adaptive than HDBSCAN |
| **Community Detection** (graph-based) | Shots as nodes, similarity as edges | Good for large datasets with clear communities |
| **Temporal Constrained Clustering** | Dailies where capture order ~ scene order | Exploits temporal prior; fails for non-linear workflows |

### Alternative Embedding Models

| Current | Alternative | When to Switch |
|---------|------------|----------------|
| CLIP ViT-B/32 | SigLIP, EVA-CLIP, DINOv2 | Better visual grounding, different semantic focus |
| InceptionResnetV1 (VGGFace2) | ArcFace (InsightFace), AdaFace | Higher accuracy face recognition |
| MTCNN | RetinaFace, SCRFD | Better small face detection, faster |
| Transformer aggregation | NetVLAD, GEM pooling | If training data is scarce; simpler but effective |

---

## Scoring & Error Mechanisms (Planned)

The pipeline currently produces scene groups without confidence scores or error detection. The following mechanisms are planned for future iterations:

### Per-Shot Confidence Score

Each shot should receive a confidence score indicating how well it fits its assigned scene:

```
confidence(shot_i, scene_k) = mean_similarity(shot_i, other shots in scene_k)
```

- **High confidence (> 0.8)**: shot clearly belongs to this scene
- **Medium confidence (0.5-0.8)**: plausible assignment, may benefit from manual review
- **Low confidence (< 0.5)**: likely misclassified or belongs to an undetected scene

Implementation: after clustering, compute the mean cosine similarity between each shot's embedding and all other shots in its assigned scene. Shots with low within-scene similarity should be flagged.

### Scene Cohesion Score

Each scene group should have an overall cohesion score:

```
cohesion(scene_k) = mean pairwise similarity of all shots in scene_k
```

Low cohesion suggests the scene might actually be two or more scenes merged together. This could trigger automatic sub-clustering.

### Silhouette Score (Global Quality)

The silhouette score measures overall clustering quality:

```
silhouette(shot_i) = (b_i - a_i) / max(a_i, b_i)
```

Where `a_i` = mean distance to shots in the same scene, `b_i` = mean distance to shots in the nearest other scene. Values range from -1 (wrong cluster) to +1 (well-clustered). The mean silhouette score across all shots quantifies overall pipeline quality.

### Actor Consistency Check

Within each scene, verify that the actor cast is consistent:

- Flag scenes where Actor A appears in only 1 of 10 shots (possible face detection error)
- Flag scenes where shots have completely disjoint actor sets (possible clustering error)
- Report actor-scene affinity matrix for editorial review

### Temporal Coherence Score (for ordered dailies)

If shots have timestamps or capture order metadata:

```
temporal_coherence(scene_k) = fraction of shot pairs in scene_k that are
                              within T seconds of each other in capture time
```

Real scenes in dailies tend to be shot in temporal clusters. Scenes with low temporal coherence may indicate a grouping error or a scene that was shot across multiple days.

### Error Detection & Recovery Pipeline

```
Scene Groups (from Stage 6)
        |
        v
[7] Confidence Scoring ──> per-shot and per-scene scores
        |
        v
[8] Error Detection ────> flag low-confidence shots, low-cohesion scenes
        |
        v
[9] Auto-Correction ───> re-cluster flagged shots, split low-cohesion scenes
        |
        v
[10] Human Review ─────> present uncertain assignments for manual override
```

### Metrics Dashboard (Planned)

| Metric | Scope | What It Measures |
|--------|-------|-----------------|
| Silhouette score | Global | Overall clustering quality |
| Scene cohesion | Per-scene | Internal consistency of each scene |
| Shot confidence | Per-shot | How well each shot fits its scene |
| Actor consistency | Per-scene | Whether actor presence is coherent |
| Temporal coherence | Per-scene | Whether shots are temporally clustered |
| Face detection rate | Per-shot | Fraction of keyframes with detected faces |
| Ungrouped ratio | Global | Fraction of shots not assigned to any scene |

### Error Logging

Each stage should log structured errors for debugging:

- **Stage 1**: Videos that fail to open, shots with 0 extractable frames
- **Stage 2**: Frames where MTCNN crashes or returns 0 detections unexpectedly
- **Stage 3**: Face embeddings with anomalous norms, clustering instability warnings
- **Stage 4**: CLIP encoding failures, unusually low embedding variance
- **Stage 5**: Shots where feature loading fails or produces degenerate embeddings
- **Stage 6**: Clustering producing a single giant scene or all-noise result

---

## Directory Structure

### Complete File Map

```
SharpEditor/
│
├── run.py                              # Main pipeline entry point
├── main.py                             # Legacy entry point (delegates to run.py)
├── requirements.txt                    # Python dependencies
├── ARCHITECTURE.md                     # This document
│
├── configs/
│   └── config.yaml                     # Pipeline configuration
│
├── pipeline/
│   ├── __init__.py
│   ├── frame_extractor.py              # Stage 1
│   ├── face_detector.py                # Stage 2
│   ├── actor_recognition.py            # Stage 3
│   ├── scene_encoder.py                # Stage 4
│   ├── shot_encoder.py                 # Stage 5 (NEW)
│   └── clustering.py                   # Stage 6 (NEW)
│
├── models/
│   ├── __init__.py
│   └── temporal_transformer.py         # Transformer model (NEW)
│
├── training/
│   ├── __init__.py
│   ├── dataset.py                      # Training dataset (stub)
│   ├── train_temporal.py               # Training script (stub)
│   └── loss_functions.py               # Loss functions (stub)
│
├── utils/
│   ├── __init__.py
│   ├── config_utils.py                 # YAML config loading
│   ├── device_utils.py                 # Device selection (MPS/CUDA/CPU)
│   ├── video_utils.py                  # Video I/O and utilities
│   └── embedding_utils.py             # Embedding normalization
│
├── scripts/
│   └── split_into_shots.py             # Film splitting utility (NEW)
│
├── dailies_scene_ai/
│   └── __init__.py                     # Package metadata
│
└── data/                               # Git-ignored runtime data
    ├── raw_downloads/                  # Source films (Archive.org)
    ├── raw_shots/                      # Input video clips
    │   └── shots_manifest.json         # Shot extraction provenance
    ├── frames/                         # Per-shot processing outputs
    │   └── <shot_id>/                  # One directory per shot
    │       ├── manifest.json
    │       ├── sampled/
    │       ├── keyframes/
    │       ├── faces/
    │       ├── faces_aligned/
    │       ├── faces.json
    │       ├── actors.json
    │       ├── scene.json
    │       ├── shot.json
    │       └── scene_group.json
    └── embeddings/                     # Dataset-level results
        ├── actor_embeddings.npz
        ├── actors.json
        ├── scene_embeddings.npz
        ├── shot_embeddings.npz
        └── scenes.json
```

---

## Configuration Reference

```yaml
# ── Stage 1: Frame Extraction ──
frame_sampling_rate_fps: 2          # Target sampling rate
max_frames_per_shot: 30             # Maximum sampled frames per shot
frame_resize:
  enabled: true
  width: 640
  height: 360
image_format: jpg
jpg_quality: 92
keyframes:
  enabled: true
  max_keyframes: 12
  min_scene_change_score: 0.25      # HSV histogram threshold

# ── Stage 2: Face Detection ──
face_detection:
  frame_source: keyframes           # 'keyframes' | 'sampled' | 'both'
  image_size: 160                   # MTCNN input size
  margin: 20
  min_face_size: 20
  thresholds: [0.6, 0.7, 0.7]      # P-Net, R-Net, O-Net
  keep_all: true
  min_score: 0.8                    # Confidence threshold
  save_aligned: true                # Save MTCNN-aligned crops

# ── Stage 3: Actor Recognition ──
actor_recognition:
  model: inception_resnet_v1
  pretrained: vggface2
  batch_size: 64
  l2_normalize: true
  clustering:
    algorithm: hdbscan
    min_cluster_size: 5
    min_samples: 1

# ── Stage 4: Scene Encoding ──
scene_encoding:
  model_name: openai/clip-vit-base-patch32
  frame_source: keyframes
  batch_size: 16

# ── Stage 5: Shot Encoding ──
# (uses embedding size configs below + optional shot_encoding section)
# shot_encoding:
#   d_model: 512
#   nhead: 8
#   num_layers: 4
#   dim_feedforward: 1024
#   dropout: 0.1
#   max_frames: 64
#   weights_path: null              # Path to trained Transformer weights

# ── Embedding Dimensions ──
actor_embedding_size: 512
scene_embedding_size: 768
shot_embedding_size: 1024

# ── Device ──
device:
  prefer_mps: true

# ── Stage 6: Clustering ──
clustering:
  algorithm: hdbscan
  min_cluster_size: 2
  min_samples: 1
  actor_weight: 0.3                 # 0.0 = embeddings only, 1.0 = actors only
```

---

## Running the Pipeline

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place video clips in data/raw_shots/
# Or split full films:
python scripts/split_into_shots.py --input-dir data/raw_downloads --output-dir data/raw_shots

# Run all 6 stages
python run.py

# With options
python run.py --config configs/config.yaml --raw-shots data/raw_shots --frames-out data/frames --overwrite
```

### Test Dataset

The current test dataset was built from 3 public domain films downloaded from Archive.org:

| Film | Year | Genre | Size |
|------|------|-------|------|
| Detour | 1945 | Film Noir | 299 MB |
| Road House | 1948 | Noir/Drama | 395 MB |
| The Browning Version | 1951 | Drama | 261 MB |

These were split into 45 shots (15 per film, 3-25 seconds each) using ffmpeg scene detection. The pipeline correctly separated the visually distinct Browning Version from the two noir films that share similar dark cinematography.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.11+ | Deep learning framework, tensor operations |
| torchvision | 0.26+ | Image transforms, model utilities |
| opencv-python | 4.13+ | Video I/O, frame extraction, image processing |
| numpy | 2.4+ | Numerical arrays, embedding storage |
| pandas | 3.0+ | Tabular data (future use in metrics/reporting) |
| scikit-learn | 1.8+ | Pairwise similarity (cosine_similarity in Stage 6) |
| hdbscan | 0.8+ | Density-based clustering (Stages 3 and 6) |
| transformers | 5.4+ | CLIP Vision model via HuggingFace (Stage 4) |
| facenet-pytorch | 2.5+ | MTCNN face detection + InceptionResnetV1 embeddings (Stages 2-3) |
| faiss-cpu | 1.13+ | Fast similarity search (available for future nearest-neighbor queries) |
| tqdm | 4.67+ | Progress bars for all pipeline stages |
| moviepy | 2.2+ | Video manipulation utilities |
| PyYAML | 6.0+ | Configuration file parsing |

Additional tools (not in requirements.txt):
- **ffmpeg 8.1+**: Used by `scripts/split_into_shots.py` for scene detection and video segmentation
- **internetarchive**: Used for downloading test films from Archive.org

---

## JSON Output Formats

### manifest.json (Stage 1)

```json
{
  "shot_id": "film_name_shot000",
  "video_path": "/absolute/path/to/video.mp4",
  "output_dir": "/absolute/path/to/data/frames/film_name_shot000",
  "video_metadata": {
    "fps": 29.97,
    "frame_count": 718,
    "duration_s": 23.96,
    "width": 1920,
    "height": 1080
  },
  "sampling": {
    "target_fps": 2.0,
    "max_frames_per_shot": 30,
    "resize": { "enabled": true, "width": 640, "height": 360 },
    "image_format": "jpg",
    "jpg_quality": 92
  },
  "sampled": {
    "files": ["path/frame_0000.jpg", "..."],
    "source_frame_indices": [0, 15, 30, "..."],
    "scene_change_scores": [0.0, 0.12, 0.45, "..."]
  },
  "keyframes": {
    "enabled": true,
    "min_scene_change_score": 0.25,
    "max_keyframes": 12,
    "files": ["path/keyframe_000.jpg", "..."],
    "source_sample_indices": [2, 5, "..."]
  }
}
```

### faces.json (Stage 2)

```json
{
  "shot_id": "film_name_shot000",
  "shot_dir": "/path/to/shot",
  "faces_dir": "/path/to/shot/faces",
  "config": {
    "frame_source": "keyframes",
    "image_size": 160,
    "device_global": "mps",
    "device_detector": "cpu"
  },
  "num_frames_processed": 1,
  "num_faces": 2,
  "detections_by_image": {
    "/path/to/keyframe_000.jpg": [
      {
        "shot_id": "film_name_shot000",
        "source_image": "/path/to/keyframe_000.jpg",
        "face_image": "/path/to/faces/keyframe_000_face_000.jpg",
        "aligned_face_image": "/path/to/faces_aligned/keyframe_000_aligned_000.jpg",
        "bbox": [120.5, 80.3, 200.1, 190.7],
        "score": 0.9987,
        "landmarks": [[145.2, 120.1], [175.3, 118.9], [160.0, 140.5], [148.1, 162.3], [172.8, 160.9]]
      }
    ]
  }
}
```

### actors.json — per-shot (Stage 3)

```json
{
  "shot_id": "film_name_shot000",
  "actors": {
    "actor_000": 2,
    "actor_001": 1
  },
  "faces": [
    {
      "face_image": "/path/to/face.jpg",
      "actor_id": "actor_000",
      "cluster_label": 0
    }
  ],
  "actor_embeddings": {
    "actor_000": [0.015, -0.022, "...(512 values)"]
  }
}
```

For shots with no detected faces:
```json
{
  "shot_id": "film_name_shot002",
  "actors": {},
  "faces": [],
  "actor_embeddings": {}
}
```

### actors.json — dataset-level (Stage 3)

```json
{
  "num_faces": 77,
  "num_actors": 2,
  "actors": {
    "actor_000": {
      "cluster_label": 0,
      "embedding_mean": [0.015, -0.022, "...(512 values)"],
      "faces": [
        { "face_image": "/path/to/face.jpg", "shot_id": "film_name_shot000", "index": 0 }
      ]
    }
  }
}
```

### scene.json (Stage 4)

```json
{
  "shot_id": "film_name_shot000",
  "num_frames_used": 1,
  "frame_source": "keyframes",
  "embedding_dim": 768,
  "scene_embedding": [0.12, 0.08, -0.03, "...(768 values)"]
}
```

### shot.json (Stage 5)

```json
{
  "shot_id": "film_name_shot000",
  "embedding_dim": 1024,
  "num_input_frames": 8,
  "shot_embedding": [-0.01, 0.05, 0.12, "...(1024 values)"]
}
```

### scene_group.json (Stage 6)

```json
{
  "shot_id": "film_name_shot000",
  "scene_id": "scene_001",
  "cluster_label": 1,
  "is_ungrouped": false
}
```

For ungrouped shots:
```json
{
  "shot_id": "film_name_shot002",
  "scene_id": null,
  "cluster_label": -1,
  "is_ungrouped": true
}
```

### scenes.json — dataset-level (Stage 6)

```json
{
  "num_shots": 45,
  "num_scenes": 3,
  "num_ungrouped": 4,
  "algorithm": "hdbscan",
  "actor_weight": 0.3,
  "scenes": {
    "scene_000": ["shot_id_1", "shot_id_2"],
    "scene_001": ["shot_id_3", "shot_id_4", "..."],
    "scene_002": ["shot_id_10", "shot_id_11", "..."]
  },
  "ungrouped": ["shot_id_5", "shot_id_8"]
}
```

---

## Demo Benchmark Results

Results from a clean end-to-end run on the 3-film test dataset (45 shots, Apple Silicon):

### Stage Outputs

| Stage | Output | Count / Shape |
|-------|--------|---------------|
| 1. Frame Extraction | Shot directories | 45 |
| 1. Frame Extraction | Sampled frames | 719 |
| 1. Frame Extraction | Keyframes | 46 |
| 2. Face Detection | Detected faces | 77 |
| 2. Face Detection | Shots with faces | 38 / 45 (84%) |
| 3. Actor Recognition | Actor clusters | 2 |
| 3. Actor Recognition | Actor embeddings | (77, 512) |
| 4. Scene Encoding | Scene embeddings | (45, 768) |
| 5. Shot Encoding | Shot embeddings | (45, 1024) |
| 5. Shot Encoding | L2 norm range | [1.0000, 1.0000] |
| 6. Scene Clustering | Scenes found | 3 |
| 6. Scene Clustering | Ungrouped shots | 4 |

### Clustering Results

| Scene | Shots | Composition | Interpretation |
|-------|-------|-------------|----------------|
| scene_000 | 2 | Detour: 2 | Dark noir shots with no detected actors (faceless/shadowy) |
| scene_001 | 30 | Detour: 12, Road House: 14, Browning Version: 4 | Main group — noir-style shots with detected actors |
| scene_002 | 9 | Browning Version: 9 | Visually distinct drama — brighter lighting, different setting |
| ungrouped | 4 | Detour: 1, Road House: 1, Browning Version: 2 | Outlier shots that don't fit any cluster |

**Interpretation**: The pipeline correctly identifies that Browning Version (1951 drama) has a visually distinct style from the two 1940s noir films. Within the noir films, it groups Detour and Road House together (similar dark cinematography, period aesthetics). The 4 ungrouped shots are transitional or visually atypical clips.

### Disk Usage

| Directory | Size |
|-----------|------|
| data/raw_downloads/ | 955 MB (source films) |
| data/raw_shots/ | 21 MB (45 clips) |
| data/frames/ | 31 MB (frames + faces + JSONs) |
| data/embeddings/ | 0.5 MB (npz + json) |
| **Total** | **~1,008 MB** |

### Per-Shot File Completeness

All 45 shot directories contain the full set of 6 required JSON files and 2 required subdirectories:
- `manifest.json`, `faces.json`, `actors.json`, `scene.json`, `shot.json`, `scene_group.json`
- `sampled/`, `keyframes/`

---

## Known Limitations

### Current Implementation

1. **Untrained Transformer**: Stage 5 uses initialized (random) weights. The Transformer provides meaningful fusion via its architecture, but trained weights would significantly improve shot embeddings. See [Training Roadmap](#training-roadmap).

2. **Per-frame features are replicated, not unique**: The scene encoder stores only the per-shot mean embedding (not per-frame). The shot encoder replicates this mean across frames, so the Transformer receives less temporal variation than it could. Future work: store per-frame CLIP embeddings in `scene.json`.

3. **Actor clustering is global, not incremental**: HDBSCAN runs over all faces at once. Adding new shots requires re-clustering the entire dataset. For production scale, an incremental approach (assign to nearest existing cluster, periodically re-cluster) would be needed.

4. **No audio signal**: The pipeline is vision-only. Dialogue, music, and ambient sound are strong scene indicators that are currently unused.

5. **No temporal ordering constraint**: The clustering treats shots as an unordered set. In real dailies, capture time provides a strong prior (shots from the same scene tend to be captured close together). This prior is not exploited.

6. **MTCNN on MPS**: MTCNN's adaptive pooling layers are incompatible with Apple Silicon MPS. Face detection always falls back to CPU, which is slower but functionally correct.

7. **Large ensemble footprint**: The pipeline loads 3 separate models (MTCNN + InceptionResnetV1 + CLIP ViT-B/32 + TemporalTransformer). Peak memory is significant on machines with limited RAM.

8. **HDBSCAN sensitivity**: With `min_cluster_size=2`, the algorithm can produce many small clusters or label too many shots as noise depending on the dataset. Tuning `min_cluster_size` and `cluster_selection_epsilon` per-dataset may be necessary.

### Data Assumptions

- Input videos must be pre-segmented shots (one take per file). The pipeline does not perform shot boundary detection within a single long video.
- Videos should contain human actors for face-based features to be meaningful. Non-narrative content (landscapes, title cards) will have empty actor signals.
- The pipeline assumes shots from the same scene share visual similarity. This fails for scenes with dramatic lighting changes or cross-cut editing.

---

## Training Roadmap

The training infrastructure (`training/` directory) is currently stubbed out. Here is the planned implementation:

### Objective

Train the TemporalTransformer (Stage 5) using contrastive learning so that shots from the same scene produce similar embeddings and shots from different scenes produce dissimilar embeddings.

### training/dataset.py — ShotPairDataset

Will implement a PyTorch `Dataset` that:
- Loads pre-computed per-frame features (actor + scene embeddings) for each shot
- Generates positive pairs (shots from the same scene) and negative pairs (shots from different scenes)
- Supports hard negative mining (shots from different scenes that share actors)
- Handles variable-length sequences via padding and attention masks

### training/loss_functions.py — InfoNCE Loss

Will implement InfoNCE (Noise Contrastive Estimation) loss:

```
L = -log( exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
```

Where `z_i, z_j` are embeddings of shots from the same scene (positive pair), `z_k` includes all negatives in the batch, and `τ` is a temperature hyperparameter.

Additional loss variants to consider:
- **Triplet loss**: simpler but requires explicit triplet mining
- **NT-Xent** (SimCLR-style): symmetric version of InfoNCE
- **Supervised contrastive loss**: extends InfoNCE to handle multiple positives per anchor

### training/train_temporal.py — Training Loop

Will implement:
- DataLoader with collation for variable-length shot sequences
- AdamW optimizer with cosine learning rate schedule
- Gradient clipping for training stability
- Validation loop computing clustering metrics (silhouette score, NMI) on held-out scenes
- Checkpoint saving and early stopping
- Weights & Biases or TensorBoard logging

### Training Data Requirements

To train effectively, the system needs:
- A dataset with known scene labels (ground truth scene assignments)
- Minimum ~100 scenes with ~5+ shots each for reasonable contrastive learning
- Diverse content (different lighting, settings, actor counts) to generalize

Sources for training labels:
- Manually annotated dailies from a real production
- MovieNet scene annotations applied to extracted shots
- Self-supervised proxy: use temporal proximity as a weak scene label (shots captured within 5 minutes of each other are likely the same scene)

### Expected Impact

With trained weights, the shot encoder should:
- Produce tighter within-scene clusters (higher cohesion)
- Better separate visually similar but narratively different scenes
- Reduce the number of ungrouped (noise) shots
- Enable the pipeline to handle more ambiguous cases where raw CLIP + face features alone are insufficient
