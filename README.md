# CustomCNN Object Detection Project
A custom implementation of a CNN-based object detection model, featuring automated training, testing, and inference pipelines using an MLOps pipeline.

## Model Card

### Model Details

- **Model Name**: CustomCNN
- **Model Type**: Object Detection
- **Architecture**: Custom CNN with dual prediction heads (classification + bounding box regression)
- **Framework**: PyTorch
- **Base Model**: Trained from scratch
- **Input Size**: 64x64 pixels
- **Version**: best_model.pth (saved per training run)

### Intended Use

**Primary Use Cases**:
- Object detection in images across 80 COCO categories
- MLOps pipeline demonstration and development

**Out-of-Scope Uses**:
- Critical safety applications without additional validation
- Production deployment without proper testing on target data
- Applications requiring high-precision detection on edge devices

### Training Data

- **Dataset**: COCO128 (subset of MS COCO dataset)
- **Description**: A small sample dataset containing 128 representative images from the COCO dataset, used for rapid prototyping and testing
- **Classes**: Standard COCO classes (80 object categories)
- **Purpose**: Quick validation and development - not intended for production-level performance

**Note**: COCO128 is a minimal dataset for testing workflows. For production models, training on the full COCO dataset or custom datasets is recommended, which we intend to add later.

### Training Configuration

```yaml
Epochs: 3
Batch Size: 60
Image Size: 64x64
Optimizer: Adam
Learning Rate: 0.0001
Weight Decay: 0.0005
Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)

Loss Functions:
- CrossEntropy (classification)
- SmoothL1 (bounding box regression)

Architecture Details:
- 3 convolutional blocks (16 → 32 → 64 filters) with BatchNorm and MaxPool
- Fully connected trunk (4096 → 256) with Dropout (0.3)
- 50 query embeddings for parallel detection heads
- Classification head: 256 → 81 (80 classes + background)
- Bounding box head: 256 → 64 → 4 (cx, cy, w, h normalised to [0,1])
- Total parameters: ~1.1M
```

### Performance Metrics

Performance metrics are computed automatically during validation:
- **val_loss**: Combined classification and bounding box regression loss
- **val_acc**: Classification accuracy on the validation set
- **Deployment threshold**: Model is registered and deployed if val_loss < 7.5

*Note: Actual metrics depend on training run completion and validation results. All runs are tracked in MLflow.*

### Hardware Requirements

**Supported Devices**:
- NVIDIA GPUs (CUDA)
- Apple Silicon (MPS)
- CPU (also the fallback)

### Limitations

1. **Training Scale**: Trained on COCO128, a minimal dataset suitable only for development/testing
2. **Generalization**: Limited ability to generalize to real-world scenarios due to minimal training data, updated later!
3. **Performance**: Not optimized for production use; performance may vary significantly on unseen data. Project is not done yet...
4. **Robustness**: May not handle edge cases, occlusions, or challenging lighting conditions effectively
5. **Bias**: Inherits any biases present in the COCO dataset
6. **No NMS**: Non-maximum suppression not yet implemented

---

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lukasfdahl/MLOpsV2.git
cd MLOpsV2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Complete Pipeline (Train → Evaluate → Deploy → Inference)

```bash
python src/main.py
```

This will:
1. Automatically detect available hardware (CUDA/MPS/CPU)
2. Train the CustomCNN model on COCO128 dataset
3. Evaluate the trained model and register it in MLflow if criteria are met
4. Deploy the model to Production in MLflow
5. Run batch inference on the dataset

## Project Structure

```
.
├── src/
│   ├── main.py                # Main orchestration script
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation + MLflow model registry
│   ├── deploy.py              # Deployment logging to MLflow
│   ├── inference.py           # Batch inference pipeline
│   ├── test.py                # Prediction visualisation
│   ├── config.py              # Config loader
│   ├── dataloader_sample.py   # Dataloader for sample dataset
│   ├── dataloader_full.py     # Dataloader for full dataset
│   └── utility/               # Hardware, training and testing helpers
├── config/
│   └── train.config.yaml      # Training configuration
├── docker/
│   └── DockerFile             # Docker container definition
├── tests/                     # Unit tests
├── data/                      # Sample dataset
├── Jenkinsfile                # CI/CD pipeline definition
└── requirements.txt           # Python dependencies
```

## Testing

A test suite is included for validation using pytest:

```bash
pytest tests/ -v
```

## Configuration

Training parameters can be modified in `config/train.config.yaml`:

- **epochs**: Number of training epochs (default: 3)
- **batch_size**: Batch size (default: 60)
- **learning_rate**: Initial learning rate (default: 0.0001)
- **weight_decay**: Weight decay (default: 0.0005)
- **use_sample_dataset**: Toggle between sample and full dataset

## MLOps Pipeline

```
git push → Jenkins → Docker build → Unit tests → Train → Evaluate → Register → Deploy
```

All training runs, metrics, models and deployments are tracked in MLflow.

## Model Outputs

**Training Outputs**:
- Best model weights: `runs/models/best_model.pth`
- Last model weights: `runs/models/last_model.pth`
- Training curves logged to MLflow

**Evaluation Outputs**:
- Prediction visualisation: `runs/models/eval_predictions.png`
- Metrics logged to MLflow: val_loss, val_acc
- Model registered in MLflow registry if val_loss < 7.5

**Inference Outputs**:
- Per-batch throughput and latency printed to console
- Results logged to MLflow under `custom_model_inference` experiment

## Acknowledgments

- MS COCO dataset for training data
- PyTorch for the deep learning framework
