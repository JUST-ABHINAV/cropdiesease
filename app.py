import streamlit as st
import numpy as np
from PIL import Image
import os
import time

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="CropScan — Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-dark:  #1a3a2a;
    --green-mid:   #2d5a3d;
    --green-soft:  #4a8c5c;
    --green-pale:  #d4edda;
    --cream:       #f7f3ec;
    --amber:       #e8a020;
    --amber-pale:  #fdf3de;
    --red-soft:    #c0392b;
    --red-pale:    #fde8e6;
    --text-dark:   #1c2b1e;
    --text-mid:    #4a5e4e;
    --text-light:  #8a9e8d;
    --border:      #dde8de;
    --card-bg:     #ffffff;
    --page-bg:     #f2f5f0;
    --radius:      14px;
    --radius-sm:   8px;
    --shadow:      0 2px 16px rgba(26,58,42,0.08);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-dark);
}
.stApp { background: var(--page-bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1100px; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--green-dark) !important; border-right: none; }
[data-testid="stSidebar"] * { color: #d4edda !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.88rem; line-height: 1.7; color: #a8ccb0 !important;
}
.sidebar-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 4px 11px;
    font-size: 0.76rem;
    color: #d4edda !important;
    margin: 3px 2px;
}
.sidebar-stat {
    background: rgba(255,255,255,0.06);
    border-radius: var(--radius-sm);
    padding: 12px 14px; margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.1);
}
.sidebar-stat .lbl { font-size: 0.72rem; color: #7aaa84 !important; text-transform: uppercase; letter-spacing: 0.08em; }
.sidebar-stat .val { font-size: 1.3rem; font-weight: 600; color: #ffffff !important; margin-top: 2px; }

/* Page header */
.page-header {
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 2rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.page-header .icon-wrap {
    width: 52px; height: 52px; background: var(--green-dark);
    border-radius: 14px; display: flex; align-items: center; justify-content: center;
    font-size: 26px; flex-shrink: 0;
}
.page-header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important; color: var(--green-dark) !important;
    margin: 0 !important; padding: 0 !important; letter-spacing: -0.01em; line-height: 1.1;
}
.page-header p { font-size: 0.9rem; color: var(--text-mid); margin: 4px 0 0; }

/* Upload zone */
[data-testid="stFileUploader"] {
    background: var(--card-bg);
    border: 2px dashed var(--border);
    border-radius: var(--radius); padding: 2rem 1.5rem;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--green-soft); background: #f7fbf8; }

/* Cards */
.card {
    background: var(--card-bg); border-radius: var(--radius);
    padding: 1.5rem; box-shadow: var(--shadow);
    border: 1px solid var(--border);
}
.card-label {
    font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--text-light); margin-bottom: 6px;
}

/* Result */
.result-crop {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; color: var(--green-dark); line-height: 1.1; margin: 0 0 4px;
}
.result-disease { font-size: 1.05rem; color: var(--text-mid); margin-bottom: 18px; }
.disease-tag {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 600;
}
.tag-healthy  { background: var(--green-pale); color: var(--green-mid); }
.tag-diseased { background: var(--red-pale);   color: var(--red-soft);  }
.tag-warning  { background: var(--amber-pale); color: #8a6010;          }

/* Confidence */
.conf-wrap { margin: 18px 0 6px; }
.conf-header {
    display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;
}
.conf-header .conf-ttl {
    font-size: 0.76rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-light);
}
.conf-header .conf-pct {
    font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: var(--green-dark);
}
.conf-track { height: 10px; background: var(--border); border-radius: 10px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 10px; }
.fill-high   { background: linear-gradient(90deg,#2d5a3d,#4a8c5c); }
.fill-medium { background: linear-gradient(90deg,#c07a10,#e8a020); }
.fill-low    { background: linear-gradient(90deg,#a0291e,#c0392b); }
.conf-hint {
    margin-top: 8px; font-size: 0.8rem; padding: 8px 12px;
    border-radius: var(--radius-sm); display: flex; align-items: center; gap: 6px;
}
.hint-high   { background: var(--green-pale); color: var(--green-mid); }
.hint-medium { background: var(--amber-pale); color: #7a5a10; }
.hint-low    { background: var(--red-pale);   color: var(--red-soft);  }

/* Top-5 predictions */
.pred-row {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0; border-bottom: 1px solid var(--border);
}
.pred-row:last-child { border-bottom: none; }
.pred-rank {
    width: 24px; height: 24px; border-radius: 50%;
    background: var(--green-pale); color: var(--green-mid);
    font-size: 0.72rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.pred-rank.r1 { background: var(--green-dark); color: #fff; }
.pred-name { flex: 1; font-size: 0.88rem; }
.pred-name .cn { font-weight: 600; color: var(--text-dark); }
.pred-name .dn { font-size: 0.78rem; color: var(--text-light); }
.pred-bar-wrap { width: 90px; }
.pred-bar-track { height: 6px; background: var(--border); border-radius: 6px; overflow: hidden; }
.pred-bar-fill  { height: 100%; border-radius: 6px; background: var(--green-soft); }
.pred-bar-fill.r1 { background: var(--green-dark); }
.pred-pct { width: 42px; text-align: right; font-size: 0.82rem; font-weight: 600; color: var(--text-mid); }

/* Tips */
.tip-grid { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1rem; }
.tip-card {
    flex: 1; min-width: 140px; background: var(--card-bg);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: 14px; font-size: 0.82rem; color: var(--text-mid); line-height: 1.5;
}
.tip-card strong { display: block; color: var(--text-dark); margin-bottom: 3px; }
.tip-icon { font-size: 1.3rem; margin-bottom: 6px; }

/* Status pill */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--green-pale); color: var(--green-mid);
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.8rem; font-weight: 600; margin-bottom: 1.5rem;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--green-soft);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.5; transform:scale(.85); }
}

.section-divider { border: none; border-top: 1px solid var(--border); margin: 1.8rem 0; }

.empty-state { text-align: center; padding: 3rem 1rem; color: var(--text-light); }
.empty-state .big-icon { font-size: 3.5rem; margin-bottom: 12px; }
.empty-state h3 { font-family:'DM Serif Display',serif; color:var(--text-mid); font-size:1.3rem; margin-bottom:8px; }
.empty-state p  { font-size:0.88rem; max-width:320px; margin:0 auto; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_class_names(folder="dataset"):
    if not os.path.isdir(folder):
        return []
    return sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])


@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists("model.pth"):
        import torch
        from torchvision import models
        n = len(load_class_names())
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, n)
        m.load_state_dict(torch.load("model.pth", map_location="cpu"))
        m.eval()
        return m, "pytorch"
    if os.path.exists("model.h5") or os.path.isdir("saved_model"):
        import tensorflow as tf
        p = "model.h5" if os.path.exists("model.h5") else "saved_model"
        return tf.keras.models.load_model(p), "tensorflow"
    return None, None


def preprocess(img: Image.Image, fw: str):
    arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    if fw == "pytorch":
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        arr = arr.transpose(2, 0, 1)[np.newaxis]
    else:
        arr = arr[np.newaxis]
    return arr


def predict(img, model, fw, classes):
    arr = preprocess(img, fw)
    if fw == "pytorch":
        import torch
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor(arr)), 1).numpy()[0]
    else:
        probs = model.predict(arr, verbose=0)[0]
    top = int(np.argmax(probs))
    name = classes[top] if classes else f"Class {top}"
    return name, float(probs[top]) * 100, probs


def split_name(raw: str):
    parts = [p.replace("_", " ").strip().title() for p in raw.replace("___", "__").split("__")]
    return parts[0], " ".join(parts[1:]) if len(parts) > 1 else "Healthy"


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🌿 CropScan")
    st.markdown("AI-powered crop disease detection for Indian agriculture.")
    st.divider()

    class_names = load_class_names()
    model, framework = load_model()

    st.markdown("### Model info")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="sidebar-stat"><div class="lbl">Classes</div><div class="val">{len(class_names) or "—"}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="sidebar-stat"><div class="lbl">Engine</div><div class="val">{framework.title() if framework else "—"}</div></div>', unsafe_allow_html=True)

    if class_names:
        st.divider()
        st.markdown("### Supported crops")
        crops = sorted({split_name(c)[0] for c in class_names})
        st.markdown(" ".join(f'<span class="sidebar-badge">{c}</span>' for c in crops), unsafe_allow_html=True)

    st.divider()
    st.markdown("### Photo tips")
    st.markdown("- 📸 Use natural daylight\n- 🍃 One leaf per photo\n- 🔍 Fill the frame\n- 🚫 Avoid blur or shadows")


# ── Main ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="icon-wrap">🌾</div>
    <div>
        <h1>Crop Disease Detector</h1>
        <p>Upload a leaf photo — get an instant AI diagnosis</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not model:
    st.error("⚠️ No model found. Add `model.pth` (PyTorch) or `model.h5` (TensorFlow) to this folder.")
    st.stop()

st.markdown('<div class="status-pill"><div class="status-dot"></div>Model ready</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a leaf image here, or click to browse",
    type=["jpg", "jpeg", "png"],
)

# ── Result display ───────────────────────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown('<div class="card-label">Input image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.caption(f"📁 {uploaded.name}  ·  {image.width}×{image.height}px")

    with right:
        with st.spinner("Analysing leaf…"):
            time.sleep(0.3)
            top_class, confidence, all_probs = predict(image, model, framework, class_names)

        crop, disease = split_name(top_class)
        is_healthy = "healthy" in disease.lower()

        # Tag
        if is_healthy:
            tag_cls, tag_txt = "tag-healthy", "Healthy"
        elif confidence < 60:
            tag_cls, tag_txt = "tag-warning", "Low confidence"
        else:
            tag_cls, tag_txt = "tag-diseased", "Disease detected"

        # Confidence colour
        if confidence >= 75:
            fill_cls, hint_cls = "fill-high",   "hint-high"
            hint_icon, hint_msg = "✓", "High confidence — reliable result."
        elif confidence >= 50:
            fill_cls, hint_cls = "fill-medium", "hint-medium"
            hint_icon, hint_msg = "⚠", "Moderate confidence — try a clearer photo."
        else:
            fill_cls, hint_cls = "fill-low",    "hint-low"
            hint_icon, hint_msg = "✕", "Low confidence — retake in better lighting."

        st.markdown(f"""
        <div class="card-label">Prediction</div>
        <div class="card">
            <span class="disease-tag {tag_cls}">{tag_txt}</span>
            <p class="result-crop" style="margin-top:10px">{crop}</p>
            <p class="result-disease">{"No disease detected" if is_healthy else disease}</p>
            <div class="conf-wrap">
                <div class="conf-header">
                    <span class="conf-ttl">Confidence</span>
                    <span class="conf-pct">{confidence:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill {fill_cls}" style="width:{confidence:.1f}%"></div>
                </div>
                <div class="conf-hint {hint_cls}">{hint_icon} &nbsp;{hint_msg}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Top-5
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Top 5 predictions</div>', unsafe_allow_html=True)

    top5 = np.argsort(all_probs)[::-1][:5]
    rows = ""
    for rank, idx in enumerate(top5, 1):
        cn = class_names[idx] if class_names else f"Class {idx}"
        cr, ds = split_name(cn)
        pct = float(all_probs[idx]) * 100
        r1 = "r1" if rank == 1 else ""
        rows += f"""
        <div class="pred-row">
            <div class="pred-rank {r1}">{rank}</div>
            <div class="pred-name">
                <div class="cn">{cr}</div>
                <div class="dn">{ds or "Healthy"}</div>
            </div>
            <div class="pred-bar-wrap">
                <div class="pred-bar-track">
                    <div class="pred-bar-fill {r1}" style="width:{int(pct)}%"></div>
                </div>
            </div>
            <div class="pred-pct">{pct:.1f}%</div>
        </div>"""

    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

    # Low-confidence tips
    if confidence < 60:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Improve your result</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-grid">
            <div class="tip-card"><div class="tip-icon">☀️</div><strong>Use daylight</strong>Natural light shows leaf colour accurately.</div>
            <div class="tip-card"><div class="tip-icon">🍃</div><strong>One leaf</strong>Isolate a single leaf against a plain background.</div>
            <div class="tip-card"><div class="tip-icon">📐</div><strong>Fill the frame</strong>Get close so the leaf fills most of the shot.</div>
            <div class="tip-card"><div class="tip-icon">📷</div><strong>Avoid blur</strong>Tap to focus before shooting.</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="big-icon">🌿</div>
        <h3>No image uploaded yet</h3>
        <p>Upload a clear photo of a crop leaf above to get an instant disease diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
