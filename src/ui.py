# ui.py
# Dead-simple Streamlit front-end for the CustomCNN inference API.
# Upload one or more images, see the top-5 predictions + latency, and a link
# to the live Grafana dashboard. Talks to the API over HTTP, so it works
# whether the API runs in the same compose network or on another host.
#
# Run locally:  streamlit run src/ui.py
# In compose:   API_URL defaults to http://inference-api:8000

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://inference-api:8000")

st.set_page_config(page_title="CustomCNN Inference", page_icon="🔎", layout="centered")
st.title("🔎 CustomCNN — image classifier")
st.caption(f"Inference API: {API_URL}")

# Health badge
try:
    h = requests.get(f"{API_URL}/health", timeout=3)
    if h.ok:
        st.success(f"API healthy · device: {h.json().get('device', '?')}")
    else:
        st.warning(f"API returned HTTP {h.status_code}")
except Exception as e:
    st.error(f"Cannot reach API at {API_URL} — {e}")

st.divider()

files = st.file_uploader(
    "Upload image(s) to classify",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if files and st.button("Predict", type="primary"):
    for f in files:
        col_img, col_res = st.columns([1, 2])
        with col_img:
            st.image(f, caption=f.name, use_container_width=True)
        with col_res:
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (f.name, f.getvalue(), f.type or "image/jpeg")},
                    timeout=30,
                )
                if resp.ok:
                    data = resp.json()
                    st.metric("Top-1 class", data["top5_classes"][0],
                              f'{data["top5_confidences"][0]:.2%} conf')
                    st.caption(f'Latency: {data["latency_ms"]} ms · device: {data["device"]}')
                    st.write("**Top-5**")
                    st.table({
                        "class": data["top5_classes"],
                        "confidence": [f"{c:.2%}" for c in data["top5_confidences"]],
                    })
                else:
                    st.error(f"HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

st.divider()
st.caption("Each prediction is recorded in Prometheus and visible on the Grafana "
           "'MLOps Inference Monitoring' dashboard (port 3000).")
