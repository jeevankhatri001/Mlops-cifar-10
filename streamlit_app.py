"""
Streamlit frontend for CIFAR-10 MLOps Pipeline
Connects to the FastAPI backend running on localhost:8000
"""
import streamlit as st
import requests
import pandas as pd
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="CIFAR-10 MLOps Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("CIFAR-10 MLOps Dashboard")
st.caption("Image classification pipeline — CNN model served via FastAPI")

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Monitoring", "📈 Drift Detection"])

# ── Tab 1: Prediction ──
with tab1:
    st.header("Image Prediction")
    st.write("Upload a 32×32 RGB image to classify it into one of 10 CIFAR-10 classes.")

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded:
        col1, col2 = st.columns(2)

        with col1:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", width=200)

        with col2:
            if st.button("Classify", type="primary"):
                with st.spinner("Running inference..."):
                    try:
                        uploaded.seek(0)
                        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                        resp = requests.post(f"{API_URL}/predict", files=files, timeout=10)

                        if resp.status_code == 200:
                            result = resp.json()

                            predicted = result.get("prediction", "N/A")
                            confidence = result.get("confidence", 0)
                            latency = result.get("latency_ms", 0)
                            drift_info = result.get("drift", {})
                            flagged = drift_info.get("flagged", False) if isinstance(drift_info, dict) else False
                            z_scores = drift_info.get("z_scores", {}) if isinstance(drift_info, dict) else {}

                            st.success(f"**Predicted class:** {predicted}")
                            st.metric("Confidence", f"{confidence:.2%}")
                            st.metric("Latency", f"{latency:.1f} ms")
                            st.metric("Drift flagged", "⚠️ Yes" if flagged else "✅ No")

                            if z_scores:
                                st.write("**Z-scores per feature:**")
                                zdf = pd.DataFrame(
                                    [(k, f"{v:.3f}", "⚠️" if abs(v) > 3.0 else "✓") for k, v in z_scores.items()],
                                    columns=["Feature", "Z-Score", "Status"]
                                )
                                st.table(zdf)

                            with st.expander("Raw API response"):
                                st.json(result)
                        else:
                            st.error(f"API returned {resp.status_code}: {resp.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to FastAPI. Is it running on port 8000?")
                    except Exception as e:
                        st.error(f"Error: {e}")

# ── Tab 2: Monitoring Stats ──
with tab2:
    st.header("Monitoring Stats")
    st.write("Aggregated prediction statistics from the FastAPI monitoring endpoint.")

    if st.button("Refresh Stats", type="primary"):
        with st.spinner("Fetching stats..."):
            try:
                resp = requests.get(f"{API_URL}/monitoring/stats", timeout=10)

                if resp.status_code == 200:
                    stats = resp.json()

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Predictions", stats.get("total_predictions", "N/A"))
                    col2.metric("Drift Flagged", stats.get("drift_flagged", "N/A"))
                    avg_conf = stats.get("avg_confidence", 0)
                    col3.metric("Avg Confidence", f"{avg_conf:.2%}" if isinstance(avg_conf, float) else avg_conf)

                    by_class = stats.get("by_class", {})
                    if by_class:
                        st.subheader("Predictions by class")
                        df = pd.DataFrame(
                            list(by_class.items()),
                            columns=["Class", "Count"]
                        ).sort_values("Count", ascending=False)
                        st.bar_chart(df.set_index("Class"))

                    with st.expander("Raw API response"):
                        st.json(stats)
                else:
                    st.error(f"API returned {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Is it running on port 8000?")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Tab 3: Drift Detection ──
with tab3:
    st.header("Drift Detection")
    st.write("Population-level data drift analysis comparing recent predictions against training baseline.")

    if st.button("Run Drift Check", type="primary"):
        with st.spinner("Checking drift..."):
            try:
                resp = requests.get(f"{API_URL}/monitoring/drift", timeout=10)

                if resp.status_code == 200:
                    drift = resp.json()

                    drift_detected = drift.get("population_drift", drift.get("drift_detected", "N/A"))
                    if isinstance(drift_detected, bool):
                        if drift_detected:
                            st.error("⚠️ Population drift detected")
                        else:
                            st.success("✅ No population drift detected")
                    else:
                        st.info(f"Drift status: {drift_detected}")

                    features = drift.get("feature_details", drift.get("features", {}))
                    if features:
                        st.subheader("Per-feature drift analysis")
                        rows = []
                        for fname, fdata in features.items():
                            if isinstance(fdata, dict):
                                bmean = fdata.get('baseline_mean', 'N/A')
                                rmean = fdata.get('recent_mean', fdata.get('current_mean', 'N/A'))
                                zscore = fdata.get('z_score', 'N/A')
                                drifted = fdata.get('drifted', False)
                                rows.append({
                                    "Feature": fname,
                                    "Baseline Mean": f"{bmean:.4f}" if isinstance(bmean, float) else str(bmean),
                                    "Current Mean": f"{rmean:.4f}" if isinstance(rmean, float) else str(rmean),
                                    "Z-Score": f"{zscore:.4f}" if isinstance(zscore, float) else str(zscore),
                                    "Drifted": "⚠️ Yes" if drifted else "✓ No"
                                })
                        if rows:
                            st.table(pd.DataFrame(rows))

                    with st.expander("Raw API response"):
                        st.json(drift)
                else:
                    st.error(f"API returned {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Is it running on port 8000?")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Sidebar ──
with st.sidebar:
    st.header("About")
    st.write("**CIFAR-10 MLOps Pipeline**")
    st.write("CMP5366 Assignment")
    st.divider()
    st.write("**Tech stack:**")
    st.write("MariaDB · Redis · Airflow")
    st.write("MLflow · FastAPI · Streamlit")
    st.write("Docker · GitHub Actions")
    st.divider()
    st.write("**Model:** 3-block CNN")
    st.write("**Accuracy:** ~77%")
    st.write("**Optimizer:** Adam (lr=0.001)")
    st.divider()

    try:
        r = requests.get(f"{API_URL}/docs", timeout=3)
        st.success("FastAPI: Connected")
    except:
        st.error("FastAPI: Not reachable")
