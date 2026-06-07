> **Note on branches:** the up-to-date model and pipeline code lives on the **`development`** branch. **Module 5** (on-device quantisation) and **Module 7** (continual learning / unlearning) are implemented on their own separate branches (**`module5`** and **`module7`**).

# DAKI4 MLOps Project — CustomCNN Object Detection

This project is an implementation for the **MLOps course at Aalborg University, 2026**. Its main goal is to teach us students how an ML system should be managed from **A to Z** and how it can be implemented in the real world. This focus has implications for model performance: the aim was **not** to build the best possible model, but to build the **infrastructure around it** — reproducible training, versioning, automated CI/CD, deployment, and monitoring.

The project covers **all course modules 1–8**, with **module 5 and module 7 maintained on their own separate branches**.

## What this project demonstrates

| Area | Implementation |
|------|----------------|
| Version control | Git (`main` + `development`), pre-commit hooks (PEP8, large files, secrets) |
| Data & model versioning | DVC with a WebDAV remote |
| CI/CD | Jenkins pipeline triggered on commit; Docker images tagged with the Git commit hash and pushed to Docker Hub |
| Scalable training | Multi-GPU DDP (`torchrun`), AMP, DeepSpeed ZeRO — run on AI-LAB via SLURM |
| Scalable inference | Dynamic INT8 quantisation, magnitude pruning + fine-tuning, batch-inference benchmark |
| Experiment tracking | MLflow (runs, metrics, model registry, model card) |
| Monitoring | Prometheus + Grafana (inference metrics, drift, host/GPU), CarbonTracker |
| Drift detection | KS-test data-drift detector (validated with artificial noise) |

## Live demo

The monitoring stack runs on the AI-LAB GPU worker. The pipeline prints the current worker IP at the end of the *Start Monitoring Stack* stage — the addresses below use the most recent worker (`172.24.198.45`); substitute the printed IP if it differs.

| Service | URL |
|---------|-----|
| Inference API (Swagger — upload an image to `/predict`) | http://172.24.198.45:8000/docs |
| Prediction UI (Streamlit) | http://172.24.198.45:8501 |
| Grafana dashboards | http://172.24.198.45:3000 *(admin / admin)* |
| Prometheus | http://172.24.198.45:9090 |
| MLflow tracking UI | http://172.24.198.45:5000 |

## The model

`CustomCNN` is a DETR-style detector: an ImageNet-pretrained **ResNet-101 backbone** for feature extraction, an adaptive pool to a fixed 8×8 grid, a fully-connected trunk, and a set of learned **query embeddings** feeding two heads — a classification head (80 COCO classes + background) and a bounding-box regression head. Inputs are resized to 64×64.

---

## Model Card

### Model details
- **Name:** CustomCNN
- **Type:** Object detection (classification + bounding-box regression)
- **Architecture:** ResNet-101 backbone (ImageNet-pretrained) → adaptive pool (8×8) → FC trunk (2048·8·8 → 1024 → 512 → 256) → 50 query embeddings → classification head (256 → 81) + bbox head (256 → 64 → 4, normalised cx,cy,w,h)
- **Framework:** PyTorch
- **Input size:** 64×64
- **Checkpoint size:** ≈ 677 MB
- **Versioning:** weights versioned with DVC; every run + version tracked in MLflow

### Intended use
- **Primary:** multi-class object detection on COCO-style images; demonstrating an end-to-end MLOps pipeline.
- **Out of scope:** safety-critical or production deployment without further validation; high-precision edge use.

### Training data
- **Dataset:** COCO128 sample (128 images, versioned in Git) for fast local runs; a larger COCO subset via DVC for full training on AI-LAB.
- **Preprocessing:** resize to 64×64, normalise to [0,1]; 80/20 train/val split.

### Final training configuration
```yaml
epochs:          50
batch_size:      1536
learning_rate:   0.00225
weight_decay:    0.0005
optimizer:       Adam
scheduler:       LinearLR warmup -> ReduceLROnPlateau
precision:       AMP (mixed precision) = ON
distributed:     DDP across 2 GPUs (torchrun), DeepSpeed ZeRO stage 0
hardware:        2x NVIDIA L4 (AI-LAB)
carbon:          tracked with CarbonTracker
loss:            classification + bounding-box regression
```

### Evaluation
- **Metrics:** `val_loss` (combined classification + bbox loss) and `val_acc` (classification accuracy).
- **Deployment threshold:** a model is promoted to *Production* in the MLflow registry only if **`val_loss < 10.0`**.
- All runs are tracked in MLflow; the model card is updated automatically by `deploy.py` on each successful run.

### Limitations & ethical considerations
- Trained on a small COCO subset → limited real-world generalisation; performance is secondary to the MLOps infrastructure.
- Bounding-box quality is not measured with mAP (classification accuracy is used as a proxy); no non-maximum suppression.
- Inherits biases present in COCO; not validated for use outside a research/course context.

---

## Quick start

```bash
git clone https://github.com/lukasfdahl/MLOpsV2.git
cd MLOpsV2
pip install -r requirements.txt
```

Local smoke run on the sample dataset (auto-detects CUDA/CPU):
```bash
python src/main.py            # train on COCO128 sample
pytest tests/                 # unit tests (with coverage)
python src/benchmark.py       # batch-inference speed test
```

The full lifecycle (build → test → push → train on AI-LAB → optimise → drift → monitor → evaluate/deploy) is automated by the **Jenkins** pipeline (`Jenkinsfile`); each stage is toggled by a build parameter.

## Project structure

```
.
├── src/
│   ├── main.py                  # training entrypoint (calls train.py)
│   ├── train.py                 # DDP / AMP / DeepSpeed ZeRO training loop
│   ├── model.py                 # CustomCNN (ResNet-101 backbone + heads)
│   ├── dataloader.py            # YOLO/COCO dataloader
│   ├── deploy.py                # evaluation + MLflow registry + model card
│   ├── serve.py                 # FastAPI inference API (+ Prometheus metrics)
│   ├── ui.py                    # Streamlit prediction UI
│   ├── inference.py             # fp32 vs int8 compression benchmark
│   ├── benchmark.py             # batch throughput/latency benchmark
│   ├── drift.py                 # KS data-drift detector
│   ├── post_training.py         # quantisation + pruning helpers
│   ├── post_training_runner.py  # post-training optimisation runner
│   └── config.py                # config loader (TRAIN_CONFIG)
├── config/                      # small_train / final_train / post_training configs
├── docker/DockerFile            # project image
├── docker-compose.monitoring.yml  # API + Prometheus + Grafana + MLflow + Streamlit
├── docker-compose.system.yml    # node-exporter + cAdvisor
├── docker-compose.gpu.yml       # dcgm-exporter (GPU metrics)
├── monitoring/                  # prometheus.yml + Grafana dashboards
├── slurm/                       # AI-LAB SLURM job scripts
├── tests/                       # unit tests
├── Jenkinsfile                  # CI/CD pipeline
└── requirements.txt
```

## MLOps pipeline

```
git push → Jenkins → Docker build (commit-hash tag) → unit tests (+coverage) → push
        → train (AI-LAB, DDP) → optimise → drift → monitoring stack → evaluate → deploy
```

Code is versioned in Git, data and weights in DVC, and every run, metric, model version and the model card in MLflow — so any deployed model is fully traceable back to its commit and configuration.

## Acknowledgments
- MS COCO dataset · PyTorch · MLflow · DVC · Prometheus / Grafana · CarbonTracker
