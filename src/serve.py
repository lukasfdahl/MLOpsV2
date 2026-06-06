# serve.py
# FastAPI inference endpoint instrumented with Prometheus metrics.
# Exposes /predict for inference and /metrics for Prometheus scraping.
#
# Run locally:  uvicorn src.serve:app --host 0.0.0.0 --port 8000
# Or via docker-compose: docker-compose -f docker-compose.monitoring.yml up

import os
import sys
import time
import io
import json
import torch
import torchvision.transforms as T
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app, REGISTRY
from prometheus_client.core import GaugeMetricFamily, InfoMetricFamily
from starlette.routing import Mount

# add project src to path when running as module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CustomCNN
from config import config

# Prometheus metrics
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["status"],        
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Max softmax confidence of the top predicted class",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
MODEL_LOADED = Gauge("model_loaded", "1 if model is loaded successfully, 0 otherwise")

# model setup
NUM_CLASSES = 80
CHECKPOINT  = os.environ.get(
    "MODEL_CHECKPOINT",
    os.path.join(config["path"]["run_base_dir"], "models", "best_model.pth"),
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Drift + model-card metrics
DRIFT_JSON      = os.path.join(config["path"]["run_base_dir"], "drift", "drift_results.json")
MODEL_CARD_PATH = os.environ.get("MODEL_CARD_PATH", "model_card.yaml")


class MlopsFileCollector:
    """Expose drift-report and model-card data as Prometheus metrics."""

    def collect(self):
        # Data drift (runs/drift/drift_results.json, written by drift.py)
        try:
            with open(DRIFT_JSON) as f:
                d = json.load(f)
            ks   = GaugeMetricFamily("drift_feature_ks_stat", "KS statistic per feature (higher = more drift)", labels=["feature"])
            pval = GaugeMetricFamily("drift_feature_p_value", "KS test p-value per feature", labels=["feature"])
            fdet = GaugeMetricFamily("drift_feature_detected", "1 if this feature drifted, else 0", labels=["feature"])
            for feat, vals in d.items():
                if feat == "summary":
                    continue
                ks.add_metric([feat], float(vals["ks_stat"]))
                pval.add_metric([feat], float(vals["p_value"]))
                fdet.add_metric([feat], 1.0 if vals["drift"] else 0.0)
            yield ks
            yield pval
            yield fdet
            s = d.get("summary", {})
            yield GaugeMetricFamily("drift_features_total",   "Total features tested for drift", value=float(s.get("n_features", 0)))
            yield GaugeMetricFamily("drift_features_drifted", "Number of features that drifted",  value=float(s.get("n_drifted", 0)))
            yield GaugeMetricFamily("drift_dataset_detected", "1 if overall dataset drift detected, else 0", value=1.0 if s.get("dataset_drift") else 0.0)
        except Exception:
            pass

        # Model card (model_card.yaml, updated by deploy.py)
        try:
            import yaml
            with open(MODEL_CARD_PATH) as f:
                card = yaml.safe_load(f) or {}
            md   = card.get("model_details", {}) or {}
            perf = card.get("performance", {}) or {}
            thr  = (card.get("evaluation", {}) or {}).get("deployment_threshold", {}) or {}
            info = InfoMetricFamily("model_card", "Deployed model card summary")
            info.add_metric([], {
                "name":               str(md.get("name", "")),
                "version":            str(md.get("version", "")),
                "framework":          str(md.get("framework", "")),
                "best_val_loss":      str(perf.get("best_val_loss")),
                "best_val_acc":       str(perf.get("best_val_acc")),
                "run_id":             str(perf.get("run_id")),
                "trained_at":         str(perf.get("trained_at")),
                "val_loss_threshold": str(thr.get("val_loss")),
            })
            yield info
            for metric_name, raw in (("model_card_best_val_loss", perf.get("best_val_loss")),
                                     ("model_card_best_val_acc",  perf.get("best_val_acc"))):
                try:
                    yield GaugeMetricFamily(metric_name, "From model card (deploy.py)", value=float(raw))
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass


REGISTRY.register(MlopsFileCollector())

_model = None

def get_model() -> CustomCNN:
    global _model
    if _model is None:
        if not os.path.isfile(CHECKPOINT):
            MODEL_LOADED.set(0)
            raise RuntimeError(
                f"No checkpoint found at '{CHECKPOINT}'. Train the model first.")
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        _model = CustomCNN(num_classes=NUM_CLASSES).to(DEVICE)
        _model.load_state_dict(ckpt["model_state"])
        _model.eval()
        MODEL_LOADED.set(1)
        print(f"Model loaded from {CHECKPOINT} (epoch {ckpt['epoch']})")
    return _model

# image preprocessing
PREPROCESS = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
])

# FastAPI app
app = FastAPI(title="CustomCNN Inference API")

# Mount Prometheus /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.on_event("startup")
def startup():
    """Pre-load model at startup so first request isn't slow."""
    try:
        get_model()
    except RuntimeError as e:
        print(f"Warning: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a JPEG/PNG image and return the top-5 predicted class indices
    and their confidence scores.
    """
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = PREPROCESS(img).unsqueeze(0).to(DEVICE)  # (1, 3, 64, 64)
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        model = get_model()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, boxes = model(tensor)   # logits: (1, 50, 80)
        latency = time.perf_counter() - t0

        # Aggregate predictions across the 50 proposals: take mean logits
        mean_logits = logits[0].mean(dim=0)             # (80,)
        probs = torch.softmax(mean_logits, dim=0)
        top5  = torch.topk(probs, k=5)

        confidence = float(top5.values[0])

        # Record metrics
        INFERENCE_LATENCY.observe(latency)
        PREDICTION_CONFIDENCE.observe(confidence)
        REQUEST_COUNT.labels(status="success").inc()

        return JSONResponse({
            "top5_classes":      top5.indices.tolist(),
            "top5_confidences":  [round(v, 4) for v in top5.values.tolist()],
            "latency_ms":        round(latency * 1000, 2),
            "device":            str(DEVICE),
        })

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
