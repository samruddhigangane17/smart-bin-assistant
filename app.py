"""
app.py — Smart Bin AI Assistant
================================
UI: warm cream / eco-green / earthy-brown theme
Nature background: SVG trees, leaves, earth elements
No blue colors — all text clearly visible
"""

from __future__ import annotations
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Smart Bin AI Assistant ♻️",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from instructions import map_label_to_waste, get_instructions
from utils.preprocessing import preprocess_pil
from utils.prediction import load_model, predict

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#F5F1E9"
CARD   = "#FDFAF3"
BORDER = "#DDD5C0"
GREEN  = "#5A9E57"
LGREEN = "#A7D7A9"
DGREEN = "#3A7A37"
BROWN  = "#8B5E3C"
LBROWN = "#C49A6C"
TEXT   = "#3B3228"
MUTED  = "#7A6A58"
CREAM2 = "#EDE8DC"
AMBER  = "#C47B2B"
RUST   = "#B85C38"

# ══════════════════════════════════════════════════════════════════════════════
#  CO₂ data
# ══════════════════════════════════════════════════════════════════════════════
CO2_SAVINGS = {
    "plastic":        0.5,
    "paper":          0.3,
    "metal":          1.2,
    "glass":          0.3,
    "organic / food": 0.1,
    "e-waste":        2.0,
    "unknown":        0.0,
}
CO2_MESSAGES = {
    "plastic":        ("You kept plastic out of the ocean 🌊", "♻️"),
    "paper":          ("You saved a tree's CO₂ absorption 🌳", "🌿"),
    "metal":          ("Metal recycling saves huge energy ⚡", "🔩"),
    "glass":          ("Glass is 100% recyclable forever ✨", "🍶"),
    "organic / food": ("Composting cuts methane emissions 🌱", "🌾"),
    "e-waste":        ("Proper disposal prevents toxic leaks 🛡️", "💚"),
    "unknown":        ("Every correct sort makes a difference 🌍", "🌍"),
}

# ══════════════════════════════════════════════════════════════════════════════
#  Nature SVG background elements (inline base64-safe URL-encoded SVGs)
# ══════════════════════════════════════════════════════════════════════════════

# Big tree bottom-left
TREE_L = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 160 260'%3E%3Cg opacity='0.18'%3E%3Crect x='72' y='140' width='16' height='120' rx='8' fill='%238B5E3C'/%3E%3Cellipse cx='80' cy='130' rx='55' ry='75' fill='%235A9E57'/%3E%3Cellipse cx='48' cy='160' rx='35' ry='48' fill='%234A8A47'/%3E%3Cellipse cx='112' cy='165' rx='33' ry='45' fill='%234A8A47'/%3E%3Cellipse cx='80' cy='80' rx='38' ry='52' fill='%236DB86A'/%3E%3C/g%3E%3C/svg%3E"

# Small tree top-right
TREE_R = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 160'%3E%3Cg opacity='0.14'%3E%3Crect x='45' y='90' width='10' height='70' rx='5' fill='%238B5E3C'/%3E%3Cellipse cx='50' cy='82' rx='36' ry='50' fill='%235A9E57'/%3E%3Cellipse cx='30' cy='100' rx='22' ry='30' fill='%234A8A47'/%3E%3Cellipse cx='70' cy='103' rx='21' ry='28' fill='%234A8A47'/%3E%3Cellipse cx='50' cy='50' rx='25' ry='34' fill='%236DB86A'/%3E%3C/g%3E%3C/svg%3E"

# Leaf cluster mid-left
LEAF_L = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 90 90'%3E%3Cg opacity='0.15'%3E%3Cpath d='M45 80 Q5 50 45 10 Q85 50 45 80Z' fill='%235A9E57'/%3E%3Cline x1='45' y1='80' x2='45' y2='40' stroke='%23A7D7A9' stroke-width='2.5'/%3E%3Cline x1='45' y1='65' x2='30' y2='52' stroke='%23A7D7A9' stroke-width='1.8'/%3E%3Cline x1='45' y1='57' x2='60' y2='45' stroke='%23A7D7A9' stroke-width='1.8'/%3E%3Cpath d='M20 70 Q0 50 20 30 Q40 50 20 70Z' fill='%236DB86A'/%3E%3Cpath d='M70 65 Q50 45 70 25 Q90 45 70 65Z' fill='%236DB86A'/%3E%3C/g%3E%3C/svg%3E"

# Earth globe mid-right
EARTH = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 180 180'%3E%3Cg opacity='0.10'%3E%3Ccircle cx='90' cy='90' r='75' fill='none' stroke='%235A9E57' stroke-width='3'/%3E%3Ccircle cx='90' cy='90' r='52' fill='none' stroke='%23A7D7A9' stroke-width='2'/%3E%3Ccircle cx='90' cy='90' r='28' fill='%23A7D7A9'/%3E%3Cpath d='M90 20 L98 55 L90 52 L82 55Z' fill='%235A9E57'/%3E%3Cpath d='M160 90 L125 98 L128 90 L125 82Z' fill='%235A9E57'/%3E%3Cpath d='M90 160 L82 125 L90 128 L98 125Z' fill='%235A9E57'/%3E%3Cpath d='M20 90 L55 82 L52 90 L55 98Z' fill='%235A9E57'/%3E%3C/g%3E%3C/svg%3E"

# Tiny leaves scattered top-center
LEAVES_TOP = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 80'%3E%3Cg opacity='0.12'%3E%3Cpath d='M30 60 Q10 35 30 10 Q50 35 30 60Z' fill='%235A9E57'/%3E%3Cpath d='M80 55 Q60 32 80 8 Q100 32 80 55Z' fill='%236DB86A'/%3E%3Cpath d='M130 58 Q110 35 130 12 Q150 35 130 58Z' fill='%235A9E57'/%3E%3Cpath d='M175 52 Q158 32 175 12 Q192 32 175 52Z' fill='%234A8A47'/%3E%3C/g%3E%3C/svg%3E"

# Small bush bottom-right
BUSH_R = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 140 90'%3E%3Cg opacity='0.16'%3E%3Cellipse cx='40' cy='60' rx='38' ry='30' fill='%235A9E57'/%3E%3Cellipse cx='90' cy='55' rx='42' ry='34' fill='%234A8A47'/%3E%3Cellipse cx='130' cy='65' rx='30' ry='24' fill='%236DB86A'/%3E%3Cellipse cx='65' cy='42' rx='28' ry='22' fill='%236DB86A'/%3E%3C/g%3E%3C/svg%3E"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Lato:wght@300;400;700&display=swap');

/* ── Base ─────────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {{
    background-color: {BG} !important;
    font-family: 'Lato', sans-serif;
    color: {TEXT};
}}
[data-testid="stHeader"]  {{ background: transparent !important; box-shadow: none !important; }}
[data-testid="stSidebar"] {{ background: {CREAM2} !important; }}
.block-container          {{ padding: 0 2rem 4rem; max-width: 1300px; position: relative; z-index: 1; }}

/* ── Nature background ────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image:
        url("{TREE_L}"),
        url("{TREE_R}"),
        url("{LEAF_L}"),
        url("{EARTH}"),
        url("{LEAVES_TOP}"),
        url("{BUSH_R}");
    background-repeat: no-repeat;
    background-position:
        left -15px bottom -5px,
        right -10px top 60px,
        left 10px center,
        right -40px bottom 120px,
        center top 0px,
        right -5px bottom -5px;
    background-size:
        200px,
        130px,
        110px,
        220px,
        260px,
        180px;
}}

/* ── Radio buttons ────────────────────────────────────────────────── */
div[role="radiogroup"] {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}}
div[role="radiogroup"] label {{
    background: {CREAM2} !important;
    border: 2px solid {BROWN} !important;
    border-radius: 999px !important;
    padding: 9px 24px !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: {BROWN} !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    min-width: 160px !important;
    text-align: center !important;
}}
div[role="radiogroup"] label:hover {{
    border-color: {GREEN} !important;
    color: {DGREEN} !important;
    background: #E8F5E8 !important;
}}
div[role="radiogroup"] label[aria-checked="true"] {{
    background: {GREEN} !important;
    border-color: {DGREEN} !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 10px rgba(90,158,87,0.35) !important;
}}
/* hide the radio dot */
div[role="radiogroup"] label > div:first-child {{ display: none !important; }}

/* ── File uploader ────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: {CARD} !important;
    border: 2.5px dashed {GREEN} !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span,
[data-testid="stFileUploaderDropzoneInstructions"] > div > small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small {{
    color: {BROWN} !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
}}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button {{
    background: {GREEN} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    padding: 7px 22px !important;
}}

/* ── Camera widget ────────────────────────────────────────────────── */
[data-testid="stCameraInput"] label,
[data-testid="stCameraInput"] span,
[data-testid="stCameraInput"] p {{
    color: {BROWN} !important;
    font-weight: 700 !important;
}}
[data-testid="stCameraInputButton"] button,
[data-testid="stCameraInput"] button {{
    background: {GREEN} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
}}

/* ── Streamlit default blue → green everywhere ────────────────────── */
a, a:visited {{ color: {DGREEN} !important; }}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {{
    color: {TEXT} !important;
}}
/* Alert / info / success boxes */
[data-testid="stAlert"] {{
    background: #F0FAF0 !important;
    border: 1.5px solid {LGREEN} !important;
    border-radius: 14px !important;
}}
[data-testid="stAlert"] * {{ color: {DGREEN} !important; font-weight: 600 !important; }}
/* Spinner */
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span {{ color: {GREEN} !important; font-weight: 700 !important; }}
/* Progress / st.success green */
[data-testid="stProgressBar"] > div > div {{ background: {GREEN} !important; }}

/* ── Popover ──────────────────────────────────────────────────────── */
[data-testid="stPopover"] button {{
    background: {CARD} !important;
    border: 2px solid {BROWN} !important;
    border-radius: 16px !important;
    color: {BROWN} !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    padding: 10px !important;
}}
[data-testid="stPopover"] button:hover {{
    border-color: {GREEN} !important;
    color: {DGREEN} !important;
    background: #EEF7EE !important;
}}

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {LGREEN}; border-radius: 99px; }}

#MainMenu, footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Session state
# ══════════════════════════════════════════════════════════════════════════════
for key, default in {
    "scan_count": 0,
    "snap_label": None,
    "snap_conf": 0.0,
    "snap_category": None,
    "snap_preds": [],
    "total_co2": 0.0,
    "last_image_hash": None,
    "category_counts": {
        "plastic": 0, "paper": 0, "metal": 0,
        "glass": 0, "organic / food": 0, "e-waste": 0, "unknown": 0,
    },
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
#  Model + prediction
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def run_prediction(pil_img: Image.Image):
    import hashlib, io as _io
    buf = _io.BytesIO()
    pil_img.convert("RGB").resize((64, 64)).save(buf, format="PNG")
    h = hashlib.md5(buf.getvalue()).hexdigest()
    if h == st.session_state.get("last_image_hash"):
        return
    st.session_state["last_image_hash"] = h
    preprocessed = preprocess_pil(pil_img)
    preds    = predict(get_model(), preprocessed, top_k=5)
    label    = preds[0]["label"]      if preds else "unknown"
    conf     = preds[0]["confidence"] if preds else 0.0
    category = map_label_to_waste(label)
    st.session_state.update({
        "snap_label": label, "snap_conf": conf,
        "snap_category": category, "snap_preds": preds,
    })
    st.session_state["scan_count"] += 1
    st.session_state["total_co2"]  += CO2_SAVINGS.get(category, 0.0)
    st.session_state["category_counts"][category] = (
        st.session_state["category_counts"].get(category, 0) + 1
    )

# ══════════════════════════════════════════════════════════════════════════════
#  UI helpers
# ══════════════════════════════════════════════════════════════════════════════
def _card(accent=None):
    a = accent or GREEN
    return (
        f'<div style="background:{CARD};border:1.5px solid {BORDER};border-radius:20px;'
        f'padding:1.4rem 1.6rem;margin-bottom:1rem;position:relative;overflow:hidden;'
        f'box-shadow:0 3px 14px rgba(90,158,87,0.09);">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:4px;'
        f'background:linear-gradient(90deg,{a},{LGREEN},transparent);'
        f'border-radius:20px 20px 0 0;"></div>'
    )

_end = "</div>"

def _lbl(txt):
    return (
        f'<div style="font-size:0.68rem;letter-spacing:0.16em;text-transform:uppercase;'
        f'color:{MUTED};margin-bottom:0.5rem;display:flex;align-items:center;gap:8px;">'
        f'{txt}<span style="flex:1;height:1px;background:{BORDER};display:block;"></span></div>'
    )

# ══════════════════════════════════════════════════════════════════════════════
#  Render functions
# ══════════════════════════════════════════════════════════════════════════════
def render_instructions(info: dict, category: str):
    # Map blue bin to earthy teal so text is readable on cream background
    raw_color = info["bin_color"]
    # Replace any blue with our earthy green-teal
    display_color = DGREEN if raw_color in ("#2563EB", "#3b82f6", "#1d4ed8") else raw_color
    hx = display_color.lstrip("#")
    r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    bin_style = (f"background:rgba({r},{g},{b},0.12);color:{display_color};"
                 f"border:2px solid {display_color}60;font-weight:700;")

    hazard = (
        f'<span style="background:#FFF0ED;border:1px solid #F4A58A;color:{RUST};'
        f'border-radius:999px;padding:2px 12px;font-size:0.75rem;margin-left:8px;font-weight:700;">'
        f'⚠️ Handle with care</span>'
        if info.get("hazard") else ""
    )

    steps = ""
    for i, s in enumerate(info["steps"]):
        border = "none" if i == len(info["steps"]) - 1 else f"1px solid {BORDER}"
        steps += (
            f'<div style="display:flex;gap:12px;align-items:flex-start;'
            f'padding:0.6rem 0;border-bottom:{border};font-size:0.88rem;color:{TEXT};">'
            f'<span style="background:{LGREEN}50;color:{DGREEN};border-radius:8px;'
            f'padding:2px 9px;margin-top:1px;flex-shrink:0;font-weight:800;font-size:0.72rem;">'
            f'{i+1:02d}</span><span>{s}</span></div>'
        )

    st.markdown(
        _card(display_color)
        + _lbl("🗑️ Waste Category")
        + f'<div style="font-family:Nunito,sans-serif;font-size:1.5rem;font-weight:800;'
          f'color:{BROWN};margin:0.2rem 0;text-transform:capitalize;">'
          f'{info["emoji"]}&nbsp;{category.title()}{hazard}</div>'
        + f'<span style="display:inline-flex;align-items:center;gap:6px;font-size:0.84rem;'
          f'border-radius:999px;padding:5px 16px;margin-top:0.5rem;{bin_style}">🗑️ {info["bin"]}</span>'
        + f'<div style="margin-top:1.2rem;">' + _lbl("🌿 Disposal Steps") + "</div>"
        + steps
        + f'<div style="margin-top:0.9rem;background:#FFFBEB;border:1.5px solid #F6D860;'
          f'border-radius:12px;padding:0.75rem 1rem;font-size:0.84rem;color:#7A4F00;'
          f'display:flex;gap:8px;">💡 <span>{info["tip"]}</span></div>'
        + _end,
        unsafe_allow_html=True,
    )


def render_impact(category: str):
    co2 = CO2_SAVINGS.get(category, 0.0)
    if co2 == 0.0:
        return
    msg, icon = CO2_MESSAGES.get(category, ("Great recycling! 🌍", "🌍"))
    total = st.session_state["total_co2"]
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#E8F5E8,#F5F1E9);'
        f'border:1.5px solid {LGREEN};border-radius:20px;padding:1.2rem 1.6rem;'
        f'margin-bottom:1rem;box-shadow:0 3px 14px rgba(90,158,87,0.12);">'
        + _lbl("🌍 Environmental Impact")
        + f'<div style="display:flex;align-items:center;gap:16px;margin-top:0.4rem;">'
          f'<div style="font-size:2.4rem;">{icon}</div>'
          f'<div><div style="font-family:Nunito,sans-serif;font-size:1.3rem;font-weight:800;color:{DGREEN};">'
          f'+{co2:.1f} kg CO₂ saved 🌱</div>'
          f'<div style="font-size:0.85rem;color:{MUTED};margin-top:3px;">{msg}</div></div></div>'
        + f'<div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid {LGREEN}80;'
          f'display:flex;justify-content:space-between;align-items:center;">'
          f'<span style="font-size:0.8rem;color:{MUTED};">Session total</span>'
          f'<span style="font-family:Nunito,sans-serif;font-size:1rem;font-weight:800;color:{GREEN};">'
          f'🌍 {total:.2f} kg CO₂ saved</span></div>'
        + _end,
        unsafe_allow_html=True,
    )


def render_dashboard():
    counts = st.session_state["category_counts"]
    total  = st.session_state["scan_count"]
    co2    = st.session_state["total_co2"]
    if total == 0:
        return
    active = {k: v for k, v in counts.items() if v > 0}
    ICONS  = {"plastic":"♻️","paper":"📄","metal":"🔩","glass":"🍶",
               "organic / food":"🌿","e-waste":"💻","unknown":"❓"}
    COLORS = {"plastic":GREEN,"paper":AMBER,"metal":"#8A9BA8",
               "glass":LGREEN,"organic / food":DGREEN,"e-waste":RUST,"unknown":MUTED}

    bars = "".join(
        f'<div style="margin-bottom:0.8rem;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
        f'<span style="font-size:0.88rem;color:{TEXT};text-transform:capitalize;font-weight:600;">'
        f'{ICONS.get(k,"•")} {k}</span>'
        f'<span style="font-size:0.82rem;color:{MUTED};font-weight:700;">'
        f'{v} item{"s" if v!=1 else ""}</span></div>'
        f'<div style="height:7px;background:{CREAM2};border-radius:99px;overflow:hidden;">'
        f'<div style="width:{v/total*100:.0f}%;height:100%;background:{COLORS.get(k,GREEN)};'
        f'border-radius:99px;"></div></div></div>'
        for k, v in active.items()
    )

    milestone = (
        f"🏆 Amazing! You've sorted {total} items today!"   if total >= 10 else
        f"🎉 Great job! You recycled {total} items today!"   if total >= 5 else
        f"💪 You sorted {total} item{'s' if total!=1 else ''} — keep going!"
    )

    st.markdown(
        _card(BROWN)
        + _lbl("📊 Your Recycling Dashboard")
        + f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:1.2rem;">'
        + f'<div style="background:{CREAM2};border:1.5px solid {BORDER};border-radius:14px;padding:0.8rem;text-align:center;">'
          f'<div style="font-family:Nunito,sans-serif;font-size:1.7rem;font-weight:800;color:{GREEN};">{total}</div>'
          f'<div style="font-size:0.65rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.12em;">Items</div></div>'
        + f'<div style="background:{CREAM2};border:1.5px solid {BORDER};border-radius:14px;padding:0.8rem;text-align:center;">'
          f'<div style="font-family:Nunito,sans-serif;font-size:1.7rem;font-weight:800;color:{DGREEN};">{co2:.2f}</div>'
          f'<div style="font-size:0.65rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.12em;">kg CO₂</div></div>'
        + f'<div style="background:{CREAM2};border:1.5px solid {BORDER};border-radius:14px;padding:0.8rem;text-align:center;">'
          f'<div style="font-family:Nunito,sans-serif;font-size:1.7rem;font-weight:800;color:{BROWN};">{len(active)}</div>'
          f'<div style="font-size:0.65rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.12em;">Types</div></div>'
        + "</div>"
        + bars
        + f'<div style="padding:0.75rem 1rem;background:{LGREEN}35;border:1.5px solid {LGREEN};'
          f'border-radius:12px;font-size:0.88rem;color:{DGREEN};text-align:center;font-weight:700;">'
          f'{milestone}</div>'
        + _end,
        unsafe_allow_html=True,
    )


def render_results():
    label    = st.session_state["snap_label"]
    conf     = st.session_state["snap_conf"]
    category = st.session_state["snap_category"]
    preds    = st.session_state["snap_preds"]
    if not label:
        return
    info = get_instructions(category)

    # Detection card
    st.markdown(
        _card(GREEN)
        + _lbl("🔍 Detection Result")
        + f'<div style="font-size:0.78rem;color:{MUTED};margin-bottom:4px;">Top Detected Object</div>'
          f'<div style="font-family:Nunito,sans-serif;font-size:1.35rem;font-weight:800;'
          f'color:{BROWN};text-transform:capitalize;">{label.title()}</div>'
          f'<div style="display:flex;align-items:center;gap:10px;margin-top:0.8rem;">'
          f'<div style="flex:1;height:9px;background:{CREAM2};border-radius:99px;overflow:hidden;">'
          f'<div style="width:{conf*100:.1f}%;height:100%;'
          f'background:linear-gradient(90deg,{DGREEN},{LGREEN});border-radius:99px;"></div></div>'
          f'<span style="font-size:0.9rem;font-weight:800;color:{GREEN};min-width:44px;text-align:right;">'
          f'{conf*100:.1f}%</span></div>'
        + _end,
        unsafe_allow_html=True,
    )

    # Top-5
    if preds:
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:0.4rem 0;'
            f'border-bottom:{"none" if i==len(preds)-1 else f"1px solid {BORDER}"};font-size:0.84rem;">'
            f'<span style="color:{TEXT};text-transform:capitalize;font-weight:600;">{p["label"].title()}</span>'
            f'<span style="color:{MUTED};font-weight:700;">{p["confidence"]*100:.1f}%</span></div>'
            for i, p in enumerate(preds)
        )
        st.markdown(_card(LBROWN) + _lbl("📋 Top-5 Predictions") + rows + _end, unsafe_allow_html=True)

    render_instructions(info, category)
    render_impact(category)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER with nature decoration
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:linear-gradient(160deg,#E4F0E4 0%,#F0EBE0 50%,#FBF6EE 100%);
            border-radius:0 0 36px 36px;padding:3rem 2rem 2.8rem;
            text-align:center;margin-bottom:0.5rem;position:relative;overflow:hidden;
            border-bottom:2px solid {LGREEN}70;">

  <!-- decorative leaves in header corners -->
  <div style="position:absolute;top:-10px;left:-10px;font-size:4rem;opacity:0.18;
              transform:rotate(-20deg);pointer-events:none;">🌿</div>
  <div style="position:absolute;top:-5px;right:-5px;font-size:3.5rem;opacity:0.18;
              transform:rotate(15deg) scaleX(-1);pointer-events:none;">🌿</div>
  <div style="position:absolute;bottom:-15px;left:5%;font-size:5rem;opacity:0.13;
              transform:rotate(5deg);pointer-events:none;">🌳</div>
  <div style="position:absolute;bottom:-15px;right:6%;font-size:5rem;opacity:0.13;
              transform:rotate(-8deg);pointer-events:none;">🌲</div>

  <div style="display:inline-block;background:{LGREEN}50;border:2px solid {GREEN};
              border-radius:999px;padding:5px 20px;font-size:0.72rem;
              letter-spacing:0.18em;text-transform:uppercase;color:{DGREEN};
              font-weight:800;margin-bottom:1rem;">
    🇮🇳 India-Ready &nbsp;·&nbsp; AI-Powered &nbsp;·&nbsp; Eco-Friendly
  </div>

  <h1 style="font-family:Nunito,sans-serif;font-size:clamp(2rem,5vw,3.2rem);
             font-weight:800;color:{BROWN};margin:0 0 0.4rem;line-height:1.15;">
    Smart Bin AI Assistant ♻️
  </h1>
  <p style="font-family:Nunito,sans-serif;font-size:1.15rem;color:{GREEN};
            font-weight:700;margin:0 0 0.6rem;">
    Scan. Sort. Save the Planet 🌱
  </p>
  <p style="font-size:0.92rem;color:{MUTED};max-width:480px;margin:0 auto;line-height:1.7;">
    Point your camera at any waste item and get instant sorting guidance.<br>
    <span style="color:{DGREEN};font-weight:600;">Every correct sort helps the planet 💚</span>
  </p>

  <!-- stat pills in header -->
  <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:1.5rem;">
    <span style="background:{CARD};border:1.5px solid {BORDER};border-radius:999px;
                 padding:5px 16px;font-size:0.82rem;color:{BROWN};font-weight:700;">
      ♻️ 6 Waste Categories
    </span>
    <span style="background:{CARD};border:1.5px solid {BORDER};border-radius:999px;
                 padding:5px 16px;font-size:0.82rem;color:{BROWN};font-weight:700;">
      🤖 MobileNetV2 AI
    </span>
    <span style="background:{CARD};border:1.5px solid {BORDER};border-radius:999px;
                 padding:5px 16px;font-size:0.82rem;color:{BROWN};font-weight:700;">
      🌍 {st.session_state['total_co2']:.2f} kg CO₂ Saved
    </span>
    <span style="background:{CARD};border:1.5px solid {BORDER};border-radius:999px;
                 padding:5px 16px;font-size:0.82rem;color:{BROWN};font-weight:700;">
      📊 {st.session_state['scan_count']} Scans Done
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Model load ────────────────────────────────────────────────────────────────
with st.spinner("🌿 Loading AI model… just a moment!"):
    get_model()

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Main layout
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    # Mode selector card
    st.markdown(
        _card(BROWN)
        + _lbl("📷 Choose How to Scan"),
        unsafe_allow_html=True,
    )
    mode = st.radio(
        label="Input Mode",
        options=["📸 Webcam Capture", "🖼️ Upload Image"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown(_end, unsafe_allow_html=True)

    # ── Webcam ────────────────────────────────────────────────────────────────
    if mode == "📸 Webcam Capture":
        st.markdown(
            f'<div style="background:{CARD};border:1.5px solid {BORDER};'
            f'border-radius:20px;padding:1.2rem 1.6rem;margin-bottom:0.5rem;">'
            f'<div style="font-size:0.9rem;color:{BROWN};font-weight:700;margin-bottom:0.5rem;">'
            f'📷 How to use:</div>'
            f'<div style="font-size:0.86rem;color:{MUTED};line-height:1.8;">'
            f'1. Click <b style="color:{BROWN};">Take photo</b> button below<br>'
            f'2. Hold your waste item clearly in view<br>'
            f'3. Get instant disposal guidance! 🌱</div>',
            unsafe_allow_html=True,
        )
        cam_image = st.camera_input(label="Take a photo", label_visibility="collapsed")
        st.markdown(_end, unsafe_allow_html=True)

        if cam_image is not None:
            with st.spinner("🌿 Analysing your waste item…"):
                run_prediction(Image.open(cam_image))
            st.markdown(
                f'<div style="background:#EFF8EF;border:2px solid {LGREEN};border-radius:14px;'
                f'padding:0.8rem 1.1rem;font-size:0.9rem;color:{DGREEN};font-weight:700;">'
                f'✅ Detected: <b>{st.session_state["snap_label"].title()}</b> '
                f'({st.session_state["snap_conf"]*100:.1f}% confidence)<br>'
                f'<span style="font-weight:600;color:{GREEN};">You\'re helping the planet 💚</span></div>',
                unsafe_allow_html=True,
            )

    # ── Upload ────────────────────────────────────────────────────────────────
    else:
        st.markdown(
            f'<div style="background:{CARD};border:1.5px solid {BORDER};'
            f'border-radius:20px;padding:1.2rem 1.6rem;margin-bottom:0.5rem;">'
            f'<div style="font-size:0.9rem;color:{BROWN};font-weight:700;margin-bottom:0.5rem;">'
            f'🖼️ Upload a photo:</div>'
            f'<div style="font-size:0.86rem;color:{MUTED};line-height:1.8;">'
            f'Choose a clear photo of the waste item.<br>'
            f'Supported: JPG, PNG, WEBP 📁</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload waste item image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        st.markdown(_end, unsafe_allow_html=True)

        if uploaded is not None:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Your uploaded image", use_container_width=True)
            with st.spinner("🌿 Analysing your waste item…"):
                run_prediction(pil_img)
            st.markdown(
                f'<div style="background:#EFF8EF;border:2px solid {LGREEN};border-radius:14px;'
                f'padding:0.8rem 1.1rem;font-size:0.9rem;color:{DGREEN};font-weight:700;">'
                f'✅ Detected: <b>{st.session_state["snap_label"].title()}</b> '
                f'({st.session_state["snap_conf"]*100:.1f}% confidence)<br>'
                f'<span style="font-weight:600;color:{GREEN};">Every recycle counts 🌍</span></div>',
                unsafe_allow_html=True,
            )

# ── Right column ──────────────────────────────────────────────────────────────
with right_col:
    if st.session_state.get("snap_label"):
        render_results()
    else:
        st.markdown(
            f'<div style="background:{CARD};border:1.5px solid {BORDER};border-radius:20px;'
            f'padding:4rem 2rem;text-align:center;box-shadow:0 3px 14px rgba(90,158,87,0.09);">'
            f'<div style="font-size:4rem;margin-bottom:0.8rem;">🌱</div>'
            f'<div style="font-family:Nunito,sans-serif;font-size:1.2rem;font-weight:800;'
            f'color:{BROWN};margin-bottom:0.6rem;">Ready to scan!</div>'
            f'<div style="font-size:0.9rem;color:{MUTED};line-height:1.7;">'
            f'Capture from webcam or upload an image<br>to get instant disposal guidance.</div>'
            f'<div style="margin-top:1.5rem;display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">'
            f'<span style="background:{LGREEN}30;border:1px solid {LGREEN};border-radius:999px;'
            f'padding:4px 14px;font-size:0.8rem;color:{DGREEN};font-weight:700;">♻️ Plastic</span>'
            f'<span style="background:{LGREEN}30;border:1px solid {LGREEN};border-radius:999px;'
            f'padding:4px 14px;font-size:0.8rem;color:{DGREEN};font-weight:700;">📄 Paper</span>'
            f'<span style="background:{LGREEN}30;border:1px solid {LGREEN};border-radius:999px;'
            f'padding:4px 14px;font-size:0.8rem;color:{DGREEN};font-weight:700;">🔩 Metal</span>'
            f'<span style="background:{LGREEN}30;border:1px solid {LGREEN};border-radius:999px;'
            f'padding:4px 14px;font-size:0.8rem;color:{DGREEN};font-weight:700;">🍶 Glass</span>'
            f'<span style="background:{LGREEN}30;border:1px solid {LGREEN};border-radius:999px;'
            f'padding:4px 14px;font-size:0.8rem;color:{DGREEN};font-weight:700;">🌿 Organic</span>'
            f'<span style="background:#FEE2DC;border:1px solid #F4A58A;border-radius:999px;'
            f'padding:4px 14px;font-size:0.8rem;color:{RUST};font-weight:700;">💻 E-Waste</span>'
            f'</div>'
            f'<div style="margin-top:1.5rem;font-size:0.88rem;color:{GREEN};font-weight:700;">'
            f'You\'re helping the planet 💚</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
#  Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["scan_count"] > 0:
    st.markdown(f"<hr style='border:none;border-top:2px solid {BORDER};margin:2rem 0 1.5rem;'>",
                unsafe_allow_html=True)
    render_dashboard()

# ══════════════════════════════════════════════════════════════════════════════
#  Footer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<hr style='border:none;border-top:2px solid {BORDER};margin:2rem 0 1rem;'>",
            unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;padding:0.5rem 0 1rem;">
  <div style="display:flex;gap:1.5rem;flex-wrap:wrap;justify-content:center;
              align-items:center;margin-bottom:0.8rem;">
    <div style="display:flex;align-items:center;gap:7px;">
      <div style="width:13px;height:13px;background:{GREEN};border-radius:50%;"></div>
      <span style="font-size:0.83rem;color:{MUTED};"><b style="color:{TEXT};">Green Bin</b> — Dry Waste (Plastic · Paper · Metal · Glass)</span>
    </div>
    <div style="display:flex;align-items:center;gap:7px;">
      <div style="width:13px;height:13px;background:{DGREEN};border-radius:50%;"></div>
      <span style="font-size:0.83rem;color:{MUTED};"><b style="color:{TEXT};">Darker Green Bin</b> — Wet / Organic Waste</span>
    </div>
    <div style="display:flex;align-items:center;gap:7px;">
      <div style="width:13px;height:13px;background:{RUST};border-radius:50%;"></div>
      <span style="font-size:0.83rem;color:{MUTED};"><b style="color:{TEXT};">Red / Special</b> — E-Waste Centre</span>
    </div>
  </div>
  <div style="font-size:1.5rem;margin-bottom:0.3rem;">🌍 🌱 ♻️ 🌿 🌳</div>
  <p style="font-size:0.72rem;color:{MUTED};">
    Smart Bin AI Assistant &nbsp;·&nbsp; MobileNetV2 + Streamlit &nbsp;·&nbsp; No GPU Required &nbsp;·&nbsp; 🇮🇳 India-Ready
  </p>
</div>
""", unsafe_allow_html=True)
