import streamlit as st
import numpy as np
from PIL import Image
import os
import time
# in order to run the app , run in terminal -m streamlit run "app (2).py" 
# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="CropScan — Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<div style="
    background: linear-gradient(135deg,#1a3a2a,#2d5a3d);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 2rem;
">
    <h1 style="margin:0;font-size:2.2rem;">🌿 CropScan AI</h1>
    <p style="margin-top:6px;font-size:0.95rem;opacity:0.9;">
        Smart crop & disease detection powered by deep learning
    </p>
</div>
""", unsafe_allow_html=True)
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

.stApp {
    background: linear-gradient(135deg, #eef5f0 0%, #e6f2ea 100%);
}
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
    background: rgba(255,255,255,0.7);
    border: 2px dashed #4a8c5c;
    border-radius: 16px;
    padding: 2rem;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #2d5a3d;
    background: rgba(255,255,255,0.9);
    box-shadow: 0 0 20px rgba(74,140,92,0.2);
}
/* Cards */
.card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 1.6rem;
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.12);
}
.card-label {
    font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--text-light); margin-bottom: 6px;
}

/* Result */
.result-crop {
    font-size: 2.3rem;
    font-weight: 700;
    color: #1a3a2a;
    letter-spacing: -0.5px;
}

.result-disease {
    font-size: 1.2rem;
    font-weight: 500;
    color: #2d5a3d;
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

CROPS = ["coffee", "cotton", "jute", "rice", "sugarcane", "wheat"]

# ── Model loading ─────────────────────────────────────────────────────────────

def _read_classes(txt):
    with open(txt) as f:
        return [l.strip() for l in f if l.strip()]

def _load_resnet(path, n):
    import torch
    from torchvision import models as tvm
    m = tvm.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, n)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

@st.cache_resource(show_spinner=False)
def load_all_models():
    missing = [f for f in
               ["crop_model.pth", "crop_classes.txt"] +
               [f"{c}_model.pth"   for c in CROPS] +
               [f"{c}_classes.txt" for c in CROPS]
               if not os.path.exists(f)]
    if missing:
        return None, None, None, None

    crop_classes  = _read_classes("crop_classes.txt")
    crop_model    = _load_resnet("crop_model.pth", len(crop_classes))
    disease_models, disease_classes = {}, {}
    for c in CROPS:
        dc = _read_classes(f"{c}_classes.txt")
        disease_models[c]  = _load_resnet(f"{c}_model.pth", len(dc))
        disease_classes[c] = dc
    return crop_model, crop_classes, disease_models, disease_classes

# ── Inference ─────────────────────────────────────────────────────────────────

# def preprocess(img: Image.Image):
#     import torch
#     arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
#     arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
#     return torch.tensor(arr.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32)

def preprocess(img):
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(img).unsqueeze(0)

def predict_two_stage(img, crop_model, crop_classes, disease_models, disease_classes):
    import torch
    tensor = preprocess(img)

    # Stage 1 — which crop?
    with torch.no_grad():
        c_probs = torch.softmax(crop_model(tensor), 1).numpy()[0]
    crop_idx  = int(np.argmax(c_probs))
    crop_name = crop_classes[crop_idx]
    crop_conf = float(c_probs[crop_idx]) * 100

    # Stage 2 — which disease?
    d_model  = disease_models[crop_name]
    d_classes = disease_classes[crop_name]
    with torch.no_grad():
        d_probs = torch.softmax(d_model(tensor), 1).numpy()[0]
    dis_idx  = int(np.argmax(d_probs))
    dis_name = d_classes[dis_idx]
    dis_conf = float(d_probs[dis_idx]) * 100

    return crop_name, crop_conf, c_probs, crop_classes, dis_name, dis_conf, d_probs, d_classes

def fmt_disease(raw, crop):
    """'rice_brown_spot' → 'Brown Spot'"""
    label = raw.replace(f"{crop}_", "").replace("_", " ").strip().title()
    return label

def conf_style(pct):
    if pct >= 75: return "fill-high",   "hint-high",   "✓", "High confidence — reliable result."
    if pct >= 50: return "fill-medium", "hint-medium", "⚠", "Moderate confidence — try a clearer photo."
    return          "fill-low",    "hint-low",    "✕", "Low confidence — retake in better lighting."

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🌿 CropScan")
    st.markdown("Two-stage AI: detects crop, then identifies the disease.")
    st.divider()

    crop_model, crop_classes, disease_models, disease_classes = load_all_models()
    models_ok = crop_model is not None

    st.markdown("### Model info")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="sidebar-stat"><div class="lbl">Stage 1</div><div class="val">{"✓" if models_ok else "—"}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="sidebar-stat"><div class="lbl">Stage 2</div><div class="val">{"6 ✓" if models_ok else "—"}</div></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Supported crops")
    badges = " ".join(f'<span class="sidebar-badge">{c.title()}</span>' for c in CROPS)
    st.markdown(badges, unsafe_allow_html=True)
    st.divider()
    st.markdown("### Photo tips")
    st.markdown("- 📸 Use natural daylight\n- 🍃 One leaf per photo\n- 🔍 Fill the frame\n- 🚫 Avoid blur or shadows")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="icon-wrap">🌾</div>
    <div>
        <h1>Crop Disease Detector</h1>
        <p>Upload a leaf photo — the AI identifies the crop, then the disease</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error(
        "⚠️ Model files missing. Place these files in the app folder:\n\n"
        "`crop_model.pth`, `crop_classes.txt` and one `[crop]_model.pth` + "
        "`[crop]_classes.txt` for each of: coffee, cotton, jute, rice, sugarcane, wheat."
    )
    st.stop()

st.markdown('<div class="status-pill"><div class="status-dot"></div>7 models ready</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a leaf image here, or click to browse",
    type=["jpg", "jpeg", "png"],
)

# ── Results ───────────────────────────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown('<div class="card-label">Input image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.caption(f"📁 {uploaded.name}  ·  {image.width}×{image.height}px")

    with right:
        with st.spinner("Stage 1: identifying crop…"):
            time.sleep(0.2)
            crop_name, crop_conf, c_probs, crop_classes_list, \
            dis_name, dis_conf, d_probs, d_classes = \
                predict_two_stage(image, crop_model, crop_classes,
                                  disease_models, disease_classes)

        disease_label = fmt_disease(dis_name, crop_name)
        is_healthy    = "healthy" in dis_name.lower()

        if is_healthy:
            tag_cls, tag_txt = "tag-healthy",  "Healthy"
        elif dis_conf < 60:
            tag_cls, tag_txt = "tag-warning",  "Low confidence"
        else:
            tag_cls, tag_txt = "tag-diseased", "Disease detected"

        # Stage 1 mini badge
        f1, h1, i1, m1 = conf_style(crop_conf)
        f2, h2, i2, m2 = conf_style(dis_conf)

        st.markdown(f"""
        <div class="card-label">Stage 1 — Crop identified</div>
        <div class="card" style="margin-bottom:10px">
            <p class="result-crop">{crop_name.title()}</p>
            <div class="conf-wrap" style="margin-top:8px">
                <div class="conf-header">
                    <span class="conf-ttl">Confidence</span>
                    <span class="conf-pct" style="font-size:1.3rem">{crop_conf:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill {f1}" style="width:{crop_conf:.1f}%"></div>
                </div>
            </div>
        </div>

        <div class="card-label">Stage 2 — Disease identified</div>
        <div class="card">
            <span class="disease-tag {tag_cls}">{tag_txt}</span>
            <p class="result-crop" style="margin-top:8px;font-size:1.5rem">
                {"No disease detected" if is_healthy else disease_label}
            </p>
            <div class="conf-wrap">
                <div class="conf-header">
                    <span class="conf-ttl">Confidence</span>
                    <span class="conf-pct">{dis_conf:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill {f2}" style="width:{dis_conf:.1f}%"></div>
                </div>
                <div class="conf-hint {h2}">{i2} &nbsp;{m2}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Top disease predictions for the detected crop ─────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-label">All {crop_name.title()} disease probabilities</div>',
                unsafe_allow_html=True)

    top_idx = np.argsort(d_probs)[::-1]
    rows = ""
    for rank, idx in enumerate(top_idx, 1):
        label = fmt_disease(d_classes[idx], crop_name)
        pct   = float(d_probs[idx]) * 100
        r1    = "r1" if rank == 1 else ""
        rows += f"""
        <div class="pred-row">
            <div class="pred-rank {r1}">{rank}</div>
            <div class="pred-name">
                <div class="cn">{"✅ " if "healthy" in d_classes[idx] else ""}{label}</div>
            </div>
            <div class="pred-bar-wrap">
                <div class="pred-bar-track">
                    <div class="pred-bar-fill {r1}" style="width:{int(pct)}%"></div>
                </div>
            </div>
            <div class="pred-pct">{pct:.1f}%</div>
        </div>"""
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

    # Low confidence tips
    if dis_conf < 60:
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
