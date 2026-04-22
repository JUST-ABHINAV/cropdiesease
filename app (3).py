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

/* Advice cards */
.advice-card {
    background: var(--card-bg); border-radius: var(--radius);
    padding: 1.25rem 1.4rem; border: 1px solid var(--border);
    box-shadow: var(--shadow); height: 100%;
}
.advice-card.now-card    { border-top: 3px solid var(--red-soft); }
.advice-card.future-card { border-top: 3px solid var(--green-soft); }
.advice-card h4 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem; margin: 0 0 12px;
}
.advice-card.now-card h4    { color: var(--red-soft); }
.advice-card.future-card h4 { color: var(--green-mid); }
.advice-item {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 7px 0; border-bottom: 1px solid var(--border);
    font-size: 0.86rem; color: var(--text-mid); line-height: 1.5;
}
.advice-item:last-child { border-bottom: none; }
.advice-dot {
    width: 7px; height: 7px; border-radius: 50%;
    flex-shrink: 0; margin-top: 6px;
}
.now-dot    { background: var(--red-soft); }
.future-dot { background: var(--green-soft); }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────────

CROPS = ["coffee", "cotton", "jute", "rice", "sugarcane", "wheat"]

# ── Disease advice lookup ──────────────────────────────────────────────────────
DISEASE_ADVICE = {
    # Coffee
    "coffee_miner": {
        "now":     ["Remove and destroy heavily infested leaves immediately",
                    "Apply neem oil spray (5 ml/litre) on leaf undersides",
                    "Use yellow sticky traps to catch adult leaf miners"],
        "prevent": ["Inspect plants weekly, especially young shoots",
                    "Maintain proper spacing for good air circulation",
                    "Avoid excess nitrogen fertiliser — it attracts miners",
                    "Introduce parasitic wasps as natural predators if available"],
    },
    "coffee_rust": {
        "now":     ["Remove all infected leaves and burn or bury them",
                    "Apply copper-based fungicide (Bordeaux mixture) immediately",
                    "Stop wetting foliage when watering"],
        "prevent": ["Apply preventive copper fungicide before rainy season",
                    "Plant rust-resistant coffee varieties where possible",
                    "Prune plants to improve airflow and reduce humidity",
                    "Water at the base only — avoid overhead irrigation"],
    },
    # Cotton
    "cotton_aphids_edited": {
        "now":     ["Spray a strong jet of water to knock aphids off",
                    "Apply insecticidal soap or neem oil solution",
                    "Remove ant colonies nearby — ants protect aphids"],
        "prevent": ["Encourage natural predators like ladybirds and lacewings",
                    "Avoid over-fertilising with nitrogen",
                    "Monitor plants weekly from the seedling stage",
                    "Use reflective mulches to deter aphid landings"],
    },
    "cotton_army_worm_edited": {
        "now":     ["Hand-pick caterpillars and egg masses early in the morning",
                    "Apply Bacillus thuringiensis (Bt) spray — safe and effective",
                    "Use recommended insecticides if infestation is severe"],
        "prevent": ["Set up pheromone traps to monitor moth activity",
                    "Till soil after harvest to destroy pupae",
                    "Practice crop rotation to break the pest cycle",
                    "Plant early to avoid peak armyworm season"],
    },
    "cotton_bacterial_blight_edited": {
        "now":     ["Remove and destroy all infected plant material",
                    "Apply copper-based bactericide spray immediately",
                    "Avoid working in the field when plants are wet"],
        "prevent": ["Use certified disease-free, treated seeds only",
                    "Rotate crops — avoid planting cotton consecutively",
                    "Improve drainage to avoid waterlogging",
                    "Sanitise all farm tools between uses"],
    },
    "cotton_powdery_mildew_edited": {
        "now":     ["Apply sulphur-based fungicide or potassium bicarbonate spray",
                    "Remove the worst affected leaves",
                    "Improve air circulation around plants immediately"],
        "prevent": ["Avoid planting in shaded or poorly ventilated areas",
                    "Water at the base — wet foliage encourages mildew",
                    "Apply preventive sulphur sprays at season start",
                    "Choose mildew-resistant cotton varieties"],
    },
    "cotton_target_spot_edited": {
        "now":     ["Apply mancozeb or chlorothalonil fungicide to affected areas",
                    "Remove and destroy badly infected leaves",
                    "Prune to reduce canopy density and lower humidity"],
        "prevent": ["Rotate crops yearly to prevent pathogen build-up in soil",
                    "Avoid over-irrigation — keep the canopy dry",
                    "Apply preventive fungicide during high-humidity periods",
                    "Destroy crop residues after harvest"],
    },
    # Jute
    "jute_cescospora_leaf_spot": {
        "now":     ["Spray mancozeb (2 g/litre) at first sign of spots",
                    "Remove heavily infected lower leaves",
                    "Avoid splashing water between plants"],
        "prevent": ["Use healthy certified seeds",
                    "Maintain proper plant spacing for good airflow",
                    "Apply preventive fungicide during wet weather",
                    "Burn crop debris at the end of the season"],
    },
    "jute_golden_mosaic": {
        "now":     ["Uproot and destroy infected plants immediately — virus spreads fast",
                    "Control whiteflies (virus carriers) with neem oil spray",
                    "Do not compost infected plant material"],
        "prevent": ["Plant only virus-free certified seeds",
                    "Control whiteflies with yellow sticky traps",
                    "Plant at the recommended time to avoid peak whitefly season",
                    "Remove weeds around the field — they host the virus"],
    },
    # Rice
    "rice_bacterial_leaf_blight": {
        "now":     ["Drain the field immediately — stagnant water worsens blight",
                    "Stop nitrogen fertiliser application right away",
                    "Apply copper-based bactericide if available"],
        "prevent": ["Use certified resistant varieties (e.g. IR64, Swarna Sub1)",
                    "Avoid excessive nitrogen — it makes plants more susceptible",
                    "Ensure proper field drainage before planting",
                    "Treat seeds with streptomycin before sowing"],
    },
    "rice_brown_spot": {
        "now":     ["Apply tricyclazole or mancozeb fungicide immediately",
                    "Ensure the field is not water-stressed — improve irrigation",
                    "Apply potassium fertiliser to boost plant immunity"],
        "prevent": ["Maintain balanced soil nutrition, especially potassium",
                    "Use disease-free certified seeds",
                    "Treat seeds with thiram or captan before planting",
                    "Avoid drought stress — keep irrigation consistent"],
    },
    "rice_leaf_smut": {
        "now":     ["Apply propiconazole fungicide at early infection stage",
                    "Remove and destroy smutted panicles before spores spread",
                    "Avoid moving equipment between infected and healthy fields"],
        "prevent": ["Use smut-resistant rice varieties",
                    "Treat seeds with carbendazim before sowing",
                    "Avoid late planting — early crops escape peak smut infection",
                    "Practice crop rotation with non-cereal crops"],
    },
    # Sugarcane
    "sugarcane_mosaic": {
        "now":     ["Uproot and destroy infected plants immediately",
                    "Control aphid vectors with neem oil or insecticide",
                    "Do not use cuttings from infected plants as seed material"],
        "prevent": ["Plant only certified disease-free setts",
                    "Use mosaic-resistant varieties like Co 86032",
                    "Control aphids regularly throughout the season",
                    "Remove weed hosts around the field boundary"],
    },
    "sugarcane_redrot": {
        "now":     ["Remove and burn infected stalks — never leave in the field",
                    "Improve drainage immediately — avoid waterlogging",
                    "Drench soil around infected areas with carbendazim solution"],
        "prevent": ["Use resistant varieties like Co 0238 or CoJ 64",
                    "Treat seed setts in carbendazim before planting",
                    "Ensure good field drainage — red rot thrives when waterlogged",
                    "Avoid ratoon crops from infected fields"],
    },
    "sugarcane_rust": {
        "now":     ["Apply propiconazole or trifloxystrobin fungicide immediately",
                    "Remove and destroy heavily infected leaves",
                    "Avoid irrigation that wets the foliage"],
        "prevent": ["Plant rust-resistant sugarcane varieties",
                    "Apply preventive fungicide sprays during the rainy season",
                    "Maintain proper spacing to improve air circulation",
                    "Monitor crops regularly from the tillering stage"],
    },
    "sugarcane_yellow": {
        "now":     ["Test soil — yellowing is often a nitrogen or iron deficiency",
                    "Apply urea (top dressing) if nitrogen deficiency is confirmed",
                    "Control aphids and whiteflies that may carry yellow leaf virus"],
        "prevent": ["Use balanced fertilisation — soil test before every season",
                    "Plant certified virus-free seed setts",
                    "Control vector insects from early growth stage",
                    "Maintain consistent irrigation — avoid water stress"],
    },
    # Wheat
    "wheat_septoria": {
        "now":     ["Apply propiconazole or tebuconazole fungicide immediately",
                    "Remove lower infected leaves if crop is at early stage",
                    "Avoid overhead irrigation — it spreads spores"],
        "prevent": ["Use certified septoria-resistant wheat varieties",
                    "Apply preventive fungicide at the flag leaf stage",
                    "Practice crop rotation — septoria survives on wheat stubble",
                    "Bury or burn crop residues after harvest"],
    },
    "wheat_stripe_rust": {
        "now":     ["Apply tebuconazole or propiconazole fungicide immediately",
                    "Monitor the whole field — stripe rust spreads very rapidly",
                    "Report severe outbreaks to your local agricultural officer"],
        "prevent": ["Plant resistant varieties — check with your local seed supplier",
                    "Sow at the recommended time — avoid very early sowing",
                    "Apply preventive fungicide at the booting stage if rust is forecast",
                    "Avoid dense sowing — good airflow slows rust spread"],
    },
}

HEALTHY_ADVICE = {
    "now":     ["Your crop looks healthy — no action required right now",
                "Continue your regular irrigation and fertilisation schedule",
                "Keep monitoring weekly for any early signs of disease"],
    "prevent": ["Maintain proper plant spacing and field hygiene",
                "Use balanced fertilisation — avoid excess nitrogen",
                "Remove weeds regularly — they host pests and diseases",
                "Keep records of any treatments applied this season"],
}

def get_advice(dis_name_raw):
    key = dis_name_raw.lower().strip()
    if key in DISEASE_ADVICE:
        return DISEASE_ADVICE[key]["now"], DISEASE_ADVICE[key]["prevent"]
    if "healthy" in key:
        return HEALTHY_ADVICE["now"], HEALTHY_ADVICE["prevent"]
    return (
        ["Consult a local agricultural extension officer for guidance.",
         "Take a clear sample to your nearest Krishi Vigyan Kendra (KVK)."],
        ["Monitor the crop closely over the next few days.",
         "Maintain good field hygiene — remove dead plant material."],
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def _read_classes(txt):
    with open(txt) as f:
        return [l.strip() for l in f if l.strip()]

@st.cache_resource(show_spinner=False)
def load_all_models():
    """
    Load all 7 sklearn Random Forest models (.pkl files) and their
    class name lists (.txt files). Returns None values if any file is missing.
    """
    import joblib
    required = (
        ["crop_model.pkl", "crop_classes.txt"] +
        [f"{c}_model.pkl"   for c in CROPS] +
        [f"{c}_classes.txt" for c in CROPS]
    )
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None

    crop_classes  = _read_classes("crop_classes.txt")
    crop_model    = joblib.load("crop_model.pkl")

    disease_models, disease_classes = {}, {}
    for c in CROPS:
        disease_models[c]  = joblib.load(f"{c}_model.pkl")
        disease_classes[c] = _read_classes(f"{c}_classes.txt")

    return crop_model, crop_classes, disease_models, disease_classes

# ── Inference ─────────────────────────────────────────────────────────────────

def preprocess(img: Image.Image):
    """
    Replace the old PyTorch tensor conversion with sklearn feature extraction.
    Imports features.py (must be in the same folder as app.py).
    Returns a (1, 368) numpy array ready for clf.predict_proba().
    """
    from features import extract_features
    feat = extract_features(img)          # shape (368,)
    return feat.reshape(1, -1)            # shape (1, 368) — sklearn expects 2D

def predict_two_stage(img, crop_model, crop_classes, disease_models, disease_classes):
    """
    Two-stage prediction using sklearn Random Forests.
    Output format is IDENTICAL to the old PyTorch version so the UI is unchanged.

    Returns
    -------
    crop_name      : str    e.g. "rice"
    crop_conf      : float  0–100 confidence %
    c_probs        : np.ndarray  probability for each crop class
    crop_classes   : list of str
    dis_name       : str    e.g. "rice_brown_spot"
    dis_conf       : float  0–100 confidence %
    d_probs        : np.ndarray  probability for each disease class
    d_classes      : list of str
    """
    feat = preprocess(img)   # shape (1, 368)

    # ── Stage 1: which crop? ──────────────────────────────────────────────────
    # predict_proba returns shape (1, n_crops) → take row 0
    c_probs   = crop_model.predict_proba(feat)[0]
    crop_idx  = int(np.argmax(c_probs))
    crop_name = crop_classes[crop_idx]
    crop_conf = float(c_probs[crop_idx]) * 100

    # ── Stage 2: which disease within that crop? ──────────────────────────────
    d_model   = disease_models[crop_name]
    d_classes = disease_classes[crop_name]
    d_probs   = d_model.predict_proba(feat)[0]
    dis_idx   = int(np.argmax(d_probs))
    dis_name  = d_classes[dis_idx]
    dis_conf  = float(d_probs[dis_idx]) * 100

    return crop_name, crop_conf, c_probs, crop_classes, dis_name, dis_conf, d_probs, d_classes

def fmt_disease(raw, crop):
    """'rice_brown_spot' → 'Brown Spot'"""
    label = raw.replace(f"{crop}_", "").replace("_", " ").strip().title()
    return label

def conf_style(pct):
    if pct >= 65: return "fill-high",   "hint-high",   "✓", "High confidence — reliable result."
    if pct >= 40: return "fill-medium", "hint-medium", "⚠", "Moderate confidence — try a clearer photo."
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


    # ── Disease advice ────────────────────────────────────────────────────────
    if dis_conf >= 60:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        advice_header = "🌿 Healthy plant — keep it that way!" if is_healthy else f"🚨 {disease_label} detected — here's what to do"
        st.markdown(f'<div class="card-label">{advice_header}</div>', unsafe_allow_html=True)

        now_list, prevent_list = get_advice(dis_name)

        now_items = "".join(
            f'<div class="advice-item"><div class="advice-dot now-dot"></div><span>{item}</span></div>'
            for item in now_list
        )
        prevent_items = "".join(
            f'<div class="advice-item"><div class="advice-dot future-dot"></div><span>{item}</span></div>'
            for item in prevent_list
        )

        col_now, col_prev = st.columns(2, gap="medium")
        with col_now:
            st.markdown(f"""
            <div class="advice-card now-card">
                <h4>{"✅ Keep doing" if is_healthy else "⚠️ What to do now"}</h4>
                {now_items}
            </div>""", unsafe_allow_html=True)
        with col_prev:
            st.markdown(f"""
            <div class="advice-card future-card">
                <h4>🛡️ How to prevent this</h4>
                {prevent_items}
            </div>""", unsafe_allow_html=True)

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
