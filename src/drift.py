import os
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy import stats
import json

# paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "coco128_small", "images", "train2017")
OUT_DIR      = os.path.join(PROJECT_ROOT, "runs", "drift")
os.makedirs(OUT_DIR, exist_ok=True)

# Constants for feature extraction and drift detection
FEATURE_NAMES = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
RESIZE    = T.Resize((64, 64))
TO_TENSOR = T.ToTensor()

# drift threshold
P_THRESHOLD = 0.05   # p-value below this → drift detected

# For testing: simulate drift with Gaussian noise + brightness shift
def image_to_features(img: Image.Image) -> list:
    t = TO_TENSOR(RESIZE(img))
    return t.mean(dim=(1, 2)).tolist() + t.std(dim=(1, 2)).tolist()

# The main script runs a full drift detection pipeline:
def load_images(data_dir: str) -> list:
    paths = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")
    return [Image.open(p).convert("RGB") for p in paths]

# Simulate drift by adding Gaussian noise and a brightness shift to the images.
def apply_drift(images: list, noise_std: float = 0.15) -> list:
    """Gaussian noise + brightness shift to simulate production drift."""
    drifted = []
    for img in images:
        t = TO_TENSOR(img).float()
        t = (t + torch.randn_like(t) * noise_std + 0.2).clamp(0.0, 1.0)
        drifted.append(T.ToPILImage()(t))
    return drifted

# Run KS test per feature and aggregate results
def run_ks_drift(ref: np.ndarray, cur: np.ndarray) -> dict:
    """Run KS test per feature. Returns per-feature results + overall flag."""
    results = {}
    n_drifted = 0
    for i, name in enumerate(FEATURE_NAMES):
        stat, p = stats.ks_2samp(ref[:, i], cur[:, i])
        drifted = bool(p < P_THRESHOLD)
        if drifted:
            n_drifted += 1
        results[name] = {"ks_stat": float(round(stat, 4)), "p_value": float(round(p, 6)), "drift": drifted}
    results["summary"] = {
        "n_features": int(len(FEATURE_NAMES)),
        "n_drifted": int(n_drifted),
        "dataset_drift": bool(n_drifted > len(FEATURE_NAMES) / 2),
        "p_threshold": float(P_THRESHOLD),
    }
    return results

# Generate a simple HTML report with a table of results and a summary verdict.
def save_html_report(results: dict, ref: np.ndarray, cur: np.ndarray, path: str):
    """Generate a simple self-contained HTML report."""
    rows = ""
    for name in FEATURE_NAMES:
        r = results[name]
        color = "#ffcccc" if r["drift"] else "#ccffcc"
        rows += (
            f"<tr style='background:{color}'>"
            f"<td>{name}</td>"
            f"<td>{ref[:, FEATURE_NAMES.index(name)].mean():.4f}</td>"
            f"<td>{cur[:, FEATURE_NAMES.index(name)].mean():.4f}</td>"
            f"<td>{r['ks_stat']}</td>"
            f"<td>{r['p_value']}</td>"
            f"<td>{'YES' if r['drift'] else 'no'}</td>"
            f"</tr>"
        )
    s = results["summary"]
    verdict_color = "#ff4444" if s["dataset_drift"] else "#44aa44"
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Drift Report</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #333; color: white; }}
  .verdict {{ font-size: 1.4em; font-weight: bold; color: {verdict_color}; }}
</style></head><body>
<h1>Data Drift Report — CustomCNN</h1>
<p>Reference: clean validation images ({len(ref)} samples)<br>
Current: noise-corrupted images (Gaussian noise σ=0.15 + brightness +0.2)</p>
<p class="verdict">Dataset drift detected: {'YES ⚠' if s['dataset_drift'] else 'NO ✓'}
  ({s['n_drifted']}/{s['n_features']} features drifted, p &lt; {s['p_threshold']})</p>
<table>
  <tr><th>Feature</th><th>Ref mean</th><th>Cur mean</th>
      <th>KS stat</th><th>p-value</th><th>Drift?</th></tr>
  {rows}
</table>
<h2>Mitigation strategy</h2>
<p>If drift is detected in production: (1) collect new samples from the drifted
distribution, (2) retrigger the Jenkins pipeline with updated data to fine-tune
the model, (3) monitor model accuracy metrics — if val_loss exceeds the
deployment threshold the deploy stage will automatically reject the new version.</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

#loooop
def main():
    print("Loading images...")
    images = load_images(DATA_DIR)
    print(f"  {len(images)} images loaded")

    ref_features = np.array([image_to_features(img) for img in images])

    print("Applying artificial drift (Gaussian noise + brightness shift)...")
    drifted = apply_drift(images)
    cur_features = np.array([image_to_features(img) for img in drifted])

    print("\nRunning KS drift tests...")
    results = run_ks_drift(ref_features, cur_features)

    # Print summary
    print(f"\n{'Feature':<12} {'KS stat':>8} {'p-value':>10} {'Drift?':>8}")
    print("-" * 42)
    for name in FEATURE_NAMES:
        r = results[name]
        print(f"{name:<12} {r['ks_stat']:>8.4f} {r['p_value']:>10.6f} {'YES ⚠':>8}" if r["drift"]
              else f"{name:<12} {r['ks_stat']:>8.4f} {r['p_value']:>10.6f} {'no':>8}")

    s = results["summary"]
    print(f"\nDataset drift detected: {'YES' if s['dataset_drift'] else 'NO'} "
          f"({s['n_drifted']}/{s['n_features']} features)")

    # Save HTML report
    report_path = os.path.join(OUT_DIR, "drift_report.html")
    save_html_report(results, ref_features, cur_features, report_path)
    print(f"\nReport saved to: {report_path}")

    # Save JSON for programmatic use (clean native types — consumed by the
    # inference API's Prometheus collector to expose drift_* metrics).
    json_path = os.path.join(OUT_DIR, "drift_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return 0 if s["dataset_drift"] else 1


if __name__ == "__main__":
    sys.exit(main())
