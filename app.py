"""
UC5 - Supervision de pompes (maintenance prédictive)
Application Streamlit pour les flottes A et B et les quatre jalons.

Arborescence attendue (les fichiers placés à côté de app.py sont aussi trouvés) :
    app.py
    data/    uc5_flotteA.npz, uc5_flotteB.npz
    models/  jalon1_mlp_rul.keras, modele_jalon2_ae.keras, modele_jalon2_cnn.keras,
             modele_cnn_jalon3.keras, modele_rul_jalon3.keras, jalon4_gru_final.h5

Lancement :  streamlit run app.py

Lecture des NPZ : les clés sont reconnues sous plusieurs noms (pompe_val, val_pompe,
pump_validation, ...). Si une partition manque (par exemple « val » dans la flotte B),
une autre partition la remplace et l'application l'indique dans la barre latérale.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Supervision des pompes",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import tensorflow as tf
    import keras
except Exception as exc:  # TensorFlow / Keras absents ou cassés
    st.error(
        "TensorFlow / Keras ne peuvent pas être importés : "
        f"{type(exc).__name__}: {exc}\n\n"
        "Installez les dépendances : pip install -r requirements.txt"
    )
    st.stop()

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIRS = [BASE_DIR / "data", BASE_DIR]
MODEL_DIRS = [BASE_DIR / "models", BASE_DIR]

RUL_MAX = 125
CRIT_LIMIT = 30      # RUL < 30  : critique
WATCH_LIMIT = 60     # RUL < 60  : à surveiller
SENSOR_COLS = [f"c{i}" for i in range(8)]
FEATURE_COLS = SENSOR_COLS + ["charge"]
N_FEATURES = len(FEATURE_COLS)

# Les modèles fournis ont été enregistrés avec Keras 3.13.x
MIN_KERAS = (3, 13)
KERAS_VERSION = tuple(int(p) for p in keras.__version__.split(".")[:2] if p.isdigit())
ST_VERSION = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
# Streamlit récent : width="stretch" ; ancien : use_container_width
STRETCH = {"width": "stretch"} if ST_VERSION >= (1, 50) else {"use_container_width": True}

keras.utils.set_random_seed(42)

# Noms de clés acceptés dans les fichiers NPZ (insensible à la casse)
SPLIT_ALIASES = {
    "train": ["train", "training", "apprentissage"],
    "val": ["val", "valid", "validation", "dev"],
    "test": ["test", "eval", "evaluation"],
}
FIELD_ALIASES = {
    "pompe": ["pompe", "pompes", "pump", "pumps", "unit", "unite", "id"],
    "cycle": ["cycle", "cycles", "time"],
    "charge": ["charge", "load", "regime"],
    "capteurs": ["capteurs", "capteur", "sensors", "sensor", "x", "features"],
    "rul": ["rul"],
    "mode": ["mode", "modes", "label", "labels"],
    "spectro": ["spectro", "spectrogramme", "spectrogram", "spectrograms", "spec"],
}

# Palette : bleu / violet pour l'interface, la couleur d'alerte est réservée à l'état de santé
INK = "#1B2559"
STEEL = "#3B63E8"
VIOLET = "#7C5CFC"
TEAL = "#19B8A6"
MUTED = "#6B7494"
LINE = "#E3E8F5"
SOFT = "#C9D3F2"
OK = "#19A57F"
WATCH = "#F0A020"
CRIT = "#E5484D"
TONE_COLOR = {"crit": CRIT, "watch": WATCH, "ok": OK}
STATE_COLOR = {"Critique": CRIT, "À surveiller": WATCH, "Normal": OK}
DISPLAY_FONT = "Poppins, Inter, Arial, sans-serif"
BODY_FONT = "Inter, Segoe UI, Arial, sans-serif"

# ---------------------------------------------------------------------
# DESIGN
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --ink:#1B2559; --blue:#3B63E8; --violet:#7C5CFC; --teal:#19B8A6;
  --muted:#6B7494; --line:#E3E8F5; --card:#FFFFFF;
  --ok:#19A57F; --watch:#F0A020; --crit:#E5484D;
  --display:'Poppins','Inter',Arial,sans-serif;
  --shadow:0 10px 30px rgba(59,99,232,.10);
  --radius:18px;
}

/* Fond : dégradé clair bleu / lavande */
.stApp {
  background:
    radial-gradient(900px 520px at 88% -8%, rgba(124,92,252,.20), transparent 62%),
    radial-gradient(800px 520px at -8% 6%, rgba(59,99,232,.14), transparent 60%),
    linear-gradient(180deg,#F8FAFF 0%,#EEF2FF 100%);
  background-attachment:fixed;
  color:var(--ink);
}
.stApp, .stApp :is(p, li, label, input, textarea, button, td, th, a, caption) {
  font-family:'Inter','Segoe UI',Arial,sans-serif;
}
.stApp :is(h1, h2, h3) { font-family:var(--display); color:var(--ink); letter-spacing:0; }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1280px; padding-top:2rem; padding-bottom:4rem; }

/* Barre latérale bleu marine */
[data-testid="stSidebar"] { background:linear-gradient(180deg,#10307A 0%,#0A1D4D 100%); border-right:0; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#AFC0F0; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.16); }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.20); border-radius:12px;
}
[data-testid="stSidebar"] [data-baseweb="select"] * { color:#FFFFFF; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill:#FFFFFF; }

.brand { display:flex; align-items:center; gap:12px; padding:6px 2px 20px; }
.brand-logo {
  width:44px; height:44px; border-radius:13px; display:grid; place-items:center;
  background:linear-gradient(135deg,#5B8CFF,#8B6CFF); color:#fff; font-size:1.35rem;
  box-shadow:0 8px 18px rgba(0,0,0,.28);
}
.brand-name { font-family:var(--display); font-size:1.08rem; font-weight:600; color:#fff; line-height:1.15; }
.brand-sub { color:#9FB3E8; font-size:.78rem; margin-top:2px; }

[data-testid="stSidebar"] [role="radiogroup"] { gap:4px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  width:100%; padding:.7rem .95rem; margin:0; border-radius:12px; transition:background .15s;
}
[data-testid="stSidebar"] [role="radiogroup"] label p { color:#DCE5FF; font-size:.93rem; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:not([data-testid="stMarkdownContainer"]) { display:none; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(255,255,255,.08); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background:linear-gradient(90deg,rgba(255,255,255,.24),rgba(255,255,255,.08));
  box-shadow:inset 4px 0 0 #8FB0FF;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color:#FFFFFF; font-weight:600; }

/* En-tête de page */
.hero {
  position:relative; overflow:hidden; background:linear-gradient(135deg,#FFFFFF 0%,#EEF1FF 100%);
  border-radius:24px; box-shadow:var(--shadow); padding:28px 34px; border:1px solid rgba(255,255,255,.8);
}
.hero::after {
  content:""; position:absolute; right:-60px; top:-70px; width:300px; height:300px; border-radius:50%;
  background:radial-gradient(circle at 40% 40%, rgba(124,92,252,.30), rgba(59,99,232,.10) 60%, transparent 70%);
}
.hero > * { position:relative; z-index:1; }
.pill {
  display:inline-block; background:rgba(59,99,232,.10); color:var(--blue); font-weight:600;
  font-size:.78rem; padding:.3rem .8rem; border-radius:999px; margin-bottom:.8rem;
}
.page-title { font-family:var(--display); font-size:2.25rem; font-weight:700; line-height:1.15; color:var(--ink); margin:0; }
.page-sub { color:var(--muted); font-size:1rem; margin:.65rem 0 0; max-width:64ch; line-height:1.55; }

.section-head { font-family:var(--display); font-size:1.28rem; font-weight:600; color:var(--ink); margin:2.2rem 0 .9rem; }
.section-head::before {
  content:""; display:inline-block; width:6px; height:1.05em; border-radius:3px; margin-right:10px;
  vertical-align:-.15em; background:linear-gradient(180deg,var(--blue),var(--violet));
}
.note {
  background:rgba(59,99,232,.07); border-radius:14px; padding:.85rem 1.1rem;
  color:var(--muted); margin:.8rem 0 1rem; max-width:80ch; line-height:1.5;
}

/* Étapes des jalons */
.stepper {
  display:flex; background:var(--card); border-radius:var(--radius); box-shadow:var(--shadow);
  padding:18px 26px 14px; margin:1.2rem 0 .4rem;
}
.step { flex:1; position:relative; text-align:center; }
.step::before { content:""; position:absolute; top:13px; left:-50%; width:100%; height:3px; background:var(--line); }
.step:first-child::before { display:none; }
.step.done::before, .step.current::before { background:linear-gradient(90deg,var(--blue),var(--violet)); }
.step .dot {
  position:relative; z-index:1; width:28px; height:28px; border-radius:50%; margin:0 auto 7px;
  display:grid; place-items:center; background:#fff; border:2px solid var(--line);
  color:var(--muted); font-size:.8rem; font-weight:700;
}
.step.done .dot { background:var(--blue); border-color:var(--blue); color:#fff; }
.step.current .dot { border-color:var(--violet); color:var(--violet); box-shadow:0 0 0 5px rgba(124,92,252,.16); }
.step .lbl { font-size:.82rem; color:var(--muted); }
.step.current .lbl { color:var(--ink); font-weight:600; }

/* Cartes de mesures */
.readouts { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:16px; margin:1.4rem 0 .6rem; }
.readout {
  position:relative; overflow:hidden; background:var(--card); border-radius:var(--radius);
  box-shadow:var(--shadow); padding:18px 22px 16px 26px;
}
.readout::before { content:""; position:absolute; left:0; top:0; bottom:0; width:6px; background:linear-gradient(180deg,var(--blue),var(--violet)); }
.readout.tone-crit::before { background:var(--crit); }
.readout.tone-watch::before { background:var(--watch); }
.readout.tone-ok::before { background:var(--ok); }
.readout .lbl { color:var(--muted); font-size:.85rem; }
.readout .val { font-family:var(--display); font-size:2.1rem; font-weight:600; line-height:1.2; color:var(--ink); }
.readout .unit { font-family:'Inter',sans-serif; font-size:.88rem; font-weight:400; color:var(--muted); margin-left:6px; }
.readout.tone-crit .val { color:var(--crit); }
.readout.tone-watch .val { color:#C77F0A; }
.readout.tone-ok .val { color:var(--ok); }

/* Verdict d'une pompe */
.verdict { background:var(--card); border-radius:var(--radius); box-shadow:var(--shadow); border-top:6px solid var(--tone); padding:22px 26px; }
.verdict .state { font-family:var(--display); font-size:2rem; font-weight:700; line-height:1.1; color:var(--tone); }
.verdict .msg { margin:.7rem 0 1rem; line-height:1.55; max-width:46ch; color:var(--ink); }
.verdict dl { display:grid; grid-template-columns:auto 1fr; gap:.4rem 1.4rem; margin:0; border-top:1px solid var(--line); padding-top:.9rem; }
.verdict dt { color:var(--muted); }
.verdict dd { margin:0; font-weight:600; text-align:right; font-variant-numeric:tabular-nums; }

/* Répartition de la flotte */
.fleetbar { display:flex; height:16px; border-radius:999px; overflow:hidden; background:var(--line); }
.fleetbar span { display:block; height:100%; }
.fleetkey { display:flex; gap:1.6rem; flex-wrap:wrap; margin:.7rem 0 0; color:var(--muted); font-size:.9rem; }
.fleetkey i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; }

/* Graphiques, tableaux, conteneurs en cartes */
[data-testid="stPlotlyChart"] { background:var(--card); border-radius:var(--radius); box-shadow:var(--shadow); padding:10px 10px 2px; }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; box-shadow:var(--shadow); border:0; }
[class*="st-key-ctrl_"] { background:var(--card); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); padding:.4rem .6rem; }
.stButton button, .stDownloadButton button {
  border-radius:999px; font-weight:600; border:0; color:#fff;
  background:linear-gradient(90deg,var(--blue),var(--violet));
}
.stButton button:hover, .stDownloadButton button:hover { color:#fff; filter:brightness(1.06); }
[data-testid="stTabs"] button { font-weight:600; }
@media (max-width:700px) { .page-title { font-size:1.8rem; } .hero { padding:22px; } }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# COMPOSANTS D'AFFICHAGE
# ---------------------------------------------------------------------
STEP_LABELS = ["Durée de vie", "Anomalies", "Modes et transfert", "Historique"]


def stepper(active):
    """Bandeau des quatre jalons : ✓ terminés, violet = jalon affiché."""
    cells = ""
    for i, label in enumerate(STEP_LABELS):
        state = "done" if i < active else ("current" if i == active else "")
        mark = "✓" if i < active else str(i + 1)
        cells += f'<div class="step {state}"><div class="dot">{mark}</div><div class="lbl">{label}</div></div>'
    st.markdown(f'<div class="stepper">{cells}</div>', unsafe_allow_html=True)


def page_title(title, subtitle, step=None):
    st.markdown(
        f'<div class="hero"><div class="pill">Étude de cas 5 · {fleet_name}</div>'
        f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )
    if step is not None:
        stepper(step)


def section(title):
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def readouts(items):
    """items : liste de (libellé, valeur, unité, ton) avec ton in {'', 'ok', 'watch', 'crit'}."""
    cells = "".join(
        f'<div class="readout tone-{tone}"><div class="lbl">{label}</div>'
        f'<div class="val">{value}<span class="unit">{unit}</span></div></div>'
        for label, value, unit, tone in items
    )
    st.markdown(f'<div class="readouts">{cells}</div>', unsafe_allow_html=True)


def show_chart(fig):
    st.plotly_chart(fig, **STRETCH)


def show_table(df):
    st.dataframe(df, hide_index=True, **STRETCH)


def chart_style(fig, height=360):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, color=MUTED, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor="#E9EEFA", zeroline=False)
    return fig


def health_state(rul):
    """Renvoie (libellé, ton, message) pour une RUL en cycles."""
    if rul < CRIT_LIMIT:
        return ("Critique", "crit",
                f"Intervention à prioriser : moins de {CRIT_LIMIT} cycles avant la défaillance.")
    if rul < WATCH_LIMIT:
        return ("À surveiller", "watch",
                f"Inspection à prévoir : entre {CRIT_LIMIT} et {WATCH_LIMIT} cycles restants.")
    return ("Normal", "ok",
            f"Aucune action requise : plus de {WATCH_LIMIT} cycles restants.")


def rul_gauge(estimate, reference=None):
    """Cadran de RUL : zones critique / à surveiller / normal, trait bleu = référence."""
    _, tone, _ = health_state(estimate)
    gauge = {
        "axis": {"range": [0, RUL_MAX], "tickvals": [0, CRIT_LIMIT, WATCH_LIMIT, RUL_MAX],
                 "tickcolor": MUTED, "tickfont": {"color": MUTED}},
        "bar": {"color": TONE_COLOR[tone], "thickness": 0.3},
        "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
        "steps": [
            {"range": [0, CRIT_LIMIT], "color": "rgba(229,72,77,.20)"},
            {"range": [CRIT_LIMIT, WATCH_LIMIT], "color": "rgba(240,160,32,.24)"},
            {"range": [WATCH_LIMIT, RUL_MAX], "color": "rgba(25,165,127,.18)"},
        ],
    }
    if reference is not None:
        gauge["threshold"] = {"line": {"color": STEEL, "width": 5}, "thickness": 0.92,
                              "value": float(reference)}
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(estimate),
        number={"suffix": " cycles", "valueformat": ".1f",
                "font": {"size": 42, "family": DISPLAY_FONT, "color": INK}},
        gauge=gauge,
    ))
    fig.update_layout(height=270, margin=dict(l=30, r=30, t=30, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family=BODY_FONT))
    return fig


def verdict_panel(estimate, reference=None, extra=None):
    label, tone, message = health_state(estimate)
    rows = ""
    if reference is not None:
        rows += f"<dt>RUL de référence</dt><dd>{reference:.0f} cycles</dd>"
        rows += f"<dt>Écart de l'estimation</dt><dd>{estimate - reference:+.1f} cycles</dd>"
    for key, value in (extra or {}).items():
        rows += f"<dt>{key}</dt><dd>{value}</dd>"
    st.markdown(
        f'<div class="verdict" style="--tone:{TONE_COLOR[tone]}">'
        f'<div class="state">{label}</div><div class="msg">{message}</div>'
        f'<dl>{rows}</dl></div>',
        unsafe_allow_html=True,
    )


def rul_result(estimate, reference, extra=None):
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        show_chart(rul_gauge(estimate, reference))
        st.caption("Le trait bleu marque la RUL de référence.")
    with right:
        verdict_panel(estimate, reference, extra)


def add_zones(fig):
    fig.add_hrect(y0=0, y1=CRIT_LIMIT, fillcolor=CRIT, opacity=.08, line_width=0)
    fig.add_hrect(y0=CRIT_LIMIT, y1=WATCH_LIMIT, fillcolor=WATCH, opacity=.10, line_width=0)


def trajectory_chart(dfp, predictions, cycle, estimate, model_label):
    reference = np.minimum(dfp.rul.values, RUL_MAX)
    fig = go.Figure()
    add_zones(fig)
    fig.add_trace(go.Scatter(x=dfp.cycle, y=reference, mode="lines", name="RUL de référence",
                             line=dict(color=INK, width=2.2)))
    fig.add_trace(go.Scatter(x=dfp.cycle, y=predictions, mode="lines", name=model_label,
                             line=dict(color=VIOLET, width=2.4, dash="dot")))
    fig.add_vline(x=cycle, line_color=MUTED, line_dash="dash", line_width=1)
    fig.add_trace(go.Scatter(x=[cycle], y=[estimate], mode="markers", name="Cycle choisi",
                             marker=dict(size=12, color=TONE_COLOR[health_state(estimate)[1]],
                                         line=dict(color="#fff", width=2))))
    fig.update_layout(xaxis_title="Cycle", yaxis_title="RUL (cycles)")
    fig.update_yaxes(range=[0, RUL_MAX + 8])
    return chart_style(fig, 380)


# ---------------------------------------------------------------------
# CHARGEMENT DES DONNEES
# ---------------------------------------------------------------------
def locate(names, dirs):
    """Premier fichier existant parmi `names`, cherché dans `dirs`."""
    if isinstance(names, str):
        names = [names]
    for name in names:
        for folder in dirs:
            candidate = folder / name
            if candidate.exists():
                return candidate
    return None


def _find_key(files, field, split):
    """Cherche la clé NPZ d'un champ pour une partition (pompe_val, val_pompe, pump_validation...)."""
    lowered = {f.lower(): f for f in files}
    for field_alias in FIELD_ALIASES[field]:
        for split_alias in SPLIT_ALIASES[split]:
            for candidate in (f"{field_alias}_{split_alias}", f"{split_alias}_{field_alias}"):
                if candidate in lowered:
                    return lowered[candidate]
    return None


def _npz_inventory(archive):
    """Liste (clé, forme, type) sans charger les tableaux en mémoire."""
    rows = []
    for key in archive.files:
        try:
            with archive.zip.open(key + ".npy") as handle:
                version = np.lib.format.read_magic(handle)
                reader = (np.lib.format.read_array_header_2_0 if version[0] >= 2
                          else np.lib.format.read_array_header_1_0)
                shape, _, dtype = reader(handle)
            rows.append({"Clé": key, "Forme": str(tuple(shape)), "Type": str(dtype)})
        except Exception:
            rows.append({"Clé": key, "Forme": "?", "Type": "?"})
    return rows


def _read_split(archive, split, notes):
    """Lit une partition. Renvoie (DataFrame, spectrogrammes) ou (None, None) si elle est absente."""
    files = archive.files
    keys = {field: _find_key(files, field, split) for field in FIELD_ALIASES}
    if keys["capteurs"] is None or keys["pompe"] is None:
        return None, None

    sensors = np.asarray(archive[keys["capteurs"]]).astype("float32")
    if sensors.ndim == 3 and sensors.shape[2] in (8, 9):
        sensors = sensors[:, -1, :]
        notes.append(f"{keys['capteurs']} est en 3 dimensions : le dernier pas de temps de chaque fenêtre est utilisé.")
    if sensors.ndim != 2:
        raise ValueError(f"{keys['capteurs']} doit avoir 2 dimensions, reçu {sensors.shape}")
    n = len(sensors)

    charge = None
    if keys["charge"] is not None:
        charge = np.asarray(archive[keys["charge"]]).reshape(-1).astype("float32")
    if sensors.shape[1] == 9 and charge is None:
        charge = sensors[:, 8].copy()
        sensors = sensors[:, :8]
        notes.append(f"{keys['capteurs']} contient 9 colonnes : la dernière est lue comme la charge.")
    if sensors.shape[1] != 8:
        raise ValueError(f"{keys['capteurs']} doit avoir 8 colonnes, reçu {sensors.shape}")

    raw_pump = np.asarray(archive[keys["pompe"]]).reshape(-1)
    pump = raw_pump.astype(int) if raw_pump.dtype.kind in "iuf" else pd.factorize(raw_pump)[0] + 1

    if keys["cycle"] is not None:
        cycle = np.asarray(archive[keys["cycle"]]).reshape(-1).astype(int)
    else:
        cycle = pd.Series(pump).groupby(pump).cumcount().to_numpy() + 1
        notes.append(f"Clé de cycle absente pour « {split} » : les cycles sont numérotés dans l'ordre du fichier.")
    if charge is None:
        charge = np.zeros(n, dtype="float32")
        notes.append(f"Clé de charge absente pour « {split} » : la charge est fixée à 0.")

    for name, array in (("pompe", pump), ("cycle", cycle), ("charge", charge)):
        if len(array) != n:
            raise ValueError(f"« {name} » ({split}) compte {len(array)} lignes, les capteurs en comptent {n}.")

    frame = pd.DataFrame({
        "pompe": pump, "cycle": cycle, "charge": charge,
        "_npz_index": np.arange(n, dtype=int),
    })
    for i, col in enumerate(SENSOR_COLS):
        frame[col] = sensors[:, i]
    if keys["rul"] is not None:
        frame["rul"] = np.asarray(archive[keys["rul"]]).reshape(-1).astype(int)
    if keys["mode"] is not None:
        mode = np.asarray(archive[keys["mode"]])
        frame["mode"] = (mode.argmax(axis=1) if mode.ndim == 2 and mode.shape[1] > 1
                         else mode.reshape(-1)).astype(int)

    spectro = None
    if keys["spectro"] is not None:
        spectro = np.asarray(archive[keys["spectro"]]).astype("float32")
        if spectro.size and float(spectro.max()) > 1.5:
            spectro = spectro / 255.0
    return frame, spectro


@st.cache_resource(show_spinner="Lecture des données de la flotte…")
def load_fleet(npz_path):
    path = Path(npz_path)
    raw, spectra, notes = {}, {}, []

    with np.load(path, allow_pickle=False) as archive:
        inventory = _npz_inventory(archive)
        for split in SPLIT_ALIASES:
            frame, spectro = _read_split(archive, split, notes)
            if frame is None:
                continue
            raw[split] = frame.sort_values(["pompe", "cycle"]).reset_index(drop=True)
            if spectro is not None:
                spectra[split] = spectro
        present_keys = ", ".join(archive.files)

    if not raw:
        raise ValueError(
            f"Aucune partition reconnue dans {path.name}. Il faut au moins une clé de pompes "
            f"et une clé de capteurs par partition (ex. pompe_val, capteurs_val). Clés présentes : {present_keys}"
        )

    def usable(name):
        return name in raw and "rul" in raw[name].columns

    # La partition « val » sert à l'affichage : elle doit contenir la RUL de référence.
    if not usable("val"):
        source = next((s for s in ("test", "train") if usable(s)), None)
        if source is None:
            raise ValueError(
                f"Aucune partition de {path.name} ne contient la RUL de référence (rul_train, rul_val ou rul_test). "
                f"Clés présentes : {present_keys}"
            )
        reason = "absente" if "val" not in raw else "sans RUL"
        notes.append(f"Partition « val » {reason} dans {path.name} : la partition « {source} » est utilisée à sa place.")
        raw["val"] = raw[source].copy()
        spectra.pop("val", None)
        if source in spectra:
            spectra["val"] = spectra[source]

    if "train" not in raw:
        raw["train"] = raw["val"].copy()
        notes.append(f"Partition « train » absente dans {path.name} : la normalisation utilise « val ».")

    # Normalisation par régime de charge, sur les cycles sains du train (RUL > 125)
    healthy = raw["train"]
    if "rul" in healthy.columns:
        healthy = healthy[healthy["rul"] > RUL_MAX]
    if healthy.empty:
        healthy = raw["train"]
    means = healthy.groupby("charge")[SENSOR_COLS].mean()
    stds = healthy.groupby("charge")[SENSOR_COLS].std().replace(0, 1).fillna(1)
    global_mean = healthy[SENSOR_COLS].mean()
    global_std = healthy[SENSOR_COLS].std().replace(0, 1).fillna(1)

    normalized = {}
    for split, frame in raw.items():
        out = frame.copy()
        for load_value, indexes in out.groupby("charge").groups.items():
            mu = means.loc[load_value] if load_value in means.index else global_mean
            sd = stds.loc[load_value] if load_value in stds.index else global_std
            out.loc[indexes, SENSOR_COLS] = (out.loc[indexes, SENSOR_COLS].values - mu.values) / sd.values
        normalized[split] = out

    return raw, normalized, spectra, notes, inventory


# ---------------------------------------------------------------------
# MODELES KERAS
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(model_path):
    try:
        model = keras.models.load_model(model_path, compile=False, safe_mode=True)
        return model, None
    except Exception as exc:
        return None, (
            f"Impossible de charger {Path(model_path).name} "
            f"(TensorFlow {tf.__version__}, Keras {keras.__version__}). "
            f"{type(exc).__name__}: {str(exc)[:300]} "
            "Les modèles ont été enregistrés avec Keras 3.13 : "
            "installez les versions de requirements.txt."
        )


def get_model(*names):
    """Charge le premier modèle trouvé parmi `names` (models/ ou dossier de app.py)."""
    path = locate(list(names), MODEL_DIRS)
    if path is None:
        st.error(f"Modèle introuvable : {names[0]}. Placez-le dans le dossier models/ à côté de app.py.")
        return None
    model, error = load_model(str(path))
    if error:
        st.error(error)
    return model


def input_shape_of(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return tuple(shape)


def rnn_kind(model):
    for layer in model.layers:
        name = layer.__class__.__name__
        if name in ("LSTM", "GRU", "SimpleRNN", "Bidirectional"):
            return name
    return "RNN"


# ---------------------------------------------------------------------
# FENETRES DE DONNEES (même logique que preprocessing.py)
# ---------------------------------------------------------------------
def flat_windows(frame, steps):
    """Une fenêtre aplatie (steps × 9) par cycle ; début de série complété en répétant le 1er cycle."""
    values = frame[FEATURE_COLS].to_numpy("float32")
    padded = np.vstack([np.repeat(values[:1], steps - 1, axis=0), values]) if steps > 1 else values
    return np.stack([padded[i:i + steps].ravel() for i in range(len(values))]).astype("float32")


def seq_windows(frame, steps):
    """Une séquence (steps, 9) par cycle ; début de série complété par des zéros."""
    values = frame[FEATURE_COLS].to_numpy("float32")
    padded = np.vstack([np.zeros((steps - 1, values.shape[1]), dtype="float32"), values])
    return np.stack([padded[i:i + steps] for i in range(len(values))]).astype("float32")


def predict_rul(model, windows):
    out = np.asarray(model.predict(windows, verbose=0, batch_size=512))
    if out.reshape(len(windows), -1).shape[1] != 1:
        raise ValueError(f"Le modèle ne renvoie pas une valeur unique par entrée : {out.shape}")
    return np.clip(out.reshape(-1) * RUL_MAX, 0, RUL_MAX)


def pump_controls(key, with_cycle=True):
    """Sélecteurs de pompe et de cycle. Renvoie (pompe, données de la pompe, cycle, position)."""
    box = {"key": f"ctrl_{key}"} if ST_VERSION >= (1, 39) else {}
    with st.container(border=True, **box):
        left, right = st.columns([1, 2.4], gap="large")
        pump = left.selectbox("Pompe", pumps, format_func=lambda p: f"Pompe {p}", key=f"{key}_pump")
        dfp = val[val.pompe == pump].sort_values("cycle").reset_index(drop=True)
        cmin, cmax = int(dfp.cycle.min()), int(dfp.cycle.max())
        if not with_cycle:
            return pump, dfp, None, None, right
        if cmin == cmax:
            cycle = cmax
            right.caption(f"Un seul cycle disponible : {cmax}.")
        else:
            cycle = right.slider("Cycle observé", cmin, cmax, cmax, key=f"{key}_{pump}_cycle")
    idx = int((dfp.cycle - cycle).abs().idxmin())  # cycle le plus proche si la série a des trous
    return pump, dfp, int(dfp.cycle.iloc[idx]), idx, right


# ---------------------------------------------------------------------
# DONNEES, FLOTTE ET NAVIGATION
# ---------------------------------------------------------------------
fleet_files = {"Flotte A": "uc5_flotteA.npz", "Flotte B": "uc5_flotteB.npz"}
fleet_paths = {name: locate(fname, DATA_DIRS) for name, fname in fleet_files.items()}
existing_fleets = [name for name, path in fleet_paths.items() if path is not None]

if not existing_fleets:
    st.error(
        "Aucun fichier de données trouvé. Placez uc5_flotteA.npz et/ou uc5_flotteB.npz "
        f"dans {BASE_DIR / 'data'} (ou à côté de app.py)."
    )
    st.stop()

MENU = [
    "🏠  Vue d'ensemble",
    "📉  Jalon 1 — Durée de vie (MLP)",
    "🔍  Jalon 2 — Anomalies (auto-encodeur)",
    "🔧  Jalon 3 — Modes et transfert",
    "📈  Jalon 4 — Historique (RNN)",
    "🗂️  Fichiers et diagnostic",
]

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">⚙</div><div>'
        '<div class="brand-name">Supervision des pompes</div>'
        '<div class="brand-sub">Maintenance prédictive · cas 5</div></div></div>',
        unsafe_allow_html=True,
    )
    fleet_name = st.selectbox("Flotte", existing_fleets)
    try:
        RAW, DATA, SPECTRA, NOTES, INVENTORY = load_fleet(str(fleet_paths[fleet_name]))
    except Exception as exc:
        st.error(f"Lecture de {fleet_name} impossible : {exc}")
        st.stop()
    page = st.radio("Navigation", MENU, label_visibility="collapsed")
    st.divider()
    st.caption(f"{fleet_name} : {DATA['val'].pompe.nunique()} pompes en validation")
    for message in NOTES:
        st.caption(f"ℹ️ {message}")
    st.caption(f"TensorFlow {tf.__version__}, Keras {keras.__version__}")
    if KERAS_VERSION < MIN_KERAS:
        st.warning(
            f"Keras {keras.__version__} est trop ancien pour relire les modèles "
            "(enregistrés avec Keras 3.13). Lancez : pip install -r requirements.txt"
        )

val = DATA["val"]
if "rul" not in val.columns:
    st.error("La colonne rul de validation est absente du fichier NPZ.")
    st.stop()
pumps = sorted(val.pompe.dropna().unique().tolist())


# ---------------------------------------------------------------------
# VUE D'ENSEMBLE
# ---------------------------------------------------------------------
if page == MENU[0]:
    page_title(f"Vue d'ensemble de la {fleet_name.lower()}",
               "État des pompes au dernier cycle observé en validation, d'après la RUL de référence.")

    latest = val.sort_values("cycle").groupby("pompe").tail(1).copy()
    latest["ref"] = np.minimum(latest.rul, RUL_MAX)
    latest["état"] = np.select(
        [latest["ref"] < CRIT_LIMIT, latest["ref"] < WATCH_LIMIT],
        ["Critique", "À surveiller"], default="Normal",
    )
    n_crit = int((latest["état"] == "Critique").sum())
    n_watch = int((latest["état"] == "À surveiller").sum())
    n_ok = int((latest["état"] == "Normal").sum())

    readouts([
        ("Pompes suivies", f"{val.pompe.nunique():,}", "", ""),
        ("Cycles de validation", f"{len(val):,}", "", ""),
        ("À surveiller", n_watch, "pompes", "watch" if n_watch else ""),
        ("Critiques", n_crit, "pompes", "crit" if n_crit else ""),
    ])

    total = max(n_crit + n_watch + n_ok, 1)
    st.markdown(
        '<div class="fleetbar">'
        f'<span style="width:{100 * n_crit / total:.2f}%;background:{CRIT}"></span>'
        f'<span style="width:{100 * n_watch / total:.2f}%;background:{WATCH}"></span>'
        f'<span style="width:{100 * n_ok / total:.2f}%;background:{OK}"></span></div>'
        '<div class="fleetkey">'
        f'<div><i style="background:{CRIT}"></i>Critique : {n_crit}</div>'
        f'<div><i style="background:{WATCH}"></i>À surveiller : {n_watch}</div>'
        f'<div><i style="background:{OK}"></i>Normal : {n_ok}</div></div>',
        unsafe_allow_html=True,
    )

    section("Pompes les plus proches de la défaillance")
    worst = latest.sort_values("ref").head(15)
    fig = go.Figure(go.Bar(
        x=worst["ref"], y=[f"Pompe {p}" for p in worst.pompe], orientation="h",
        marker_color=[STATE_COLOR[s] for s in worst["état"]],
        text=[f"{v:.0f}" for v in worst["ref"]], textposition="outside", cliponaxis=False,
    ))
    for limit in (CRIT_LIMIT, WATCH_LIMIT):
        fig.add_vline(x=limit, line_dash="dot", line_color=MUTED, line_width=1)
    fig.update_layout(xaxis_title="RUL de référence (cycles)", yaxis_title="", showlegend=False)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, RUL_MAX + 10])
    show_chart(chart_style(fig, max(260, 28 * len(worst) + 60)))

    section("Répartition de la RUL selon la charge")
    hist = val.copy()
    hist["RUL de référence"] = np.minimum(hist.rul, RUL_MAX)
    hist["Charge"] = hist.charge.astype(str)
    fig = px.histogram(hist, x="RUL de référence", color="Charge", nbins=35, barmode="overlay",
                       opacity=.8, color_discrete_sequence=[STEEL, VIOLET, TEAL, "#9DB4F5"])
    fig.update_layout(xaxis_title="RUL (cycles)", yaxis_title="Observations")
    show_chart(chart_style(fig, 320))

    section("Détail par pompe")
    table = latest[["pompe", "cycle", "charge", "ref", "état"]].rename(columns={
        "pompe": "Pompe", "cycle": "Dernier cycle", "charge": "Charge",
        "ref": "RUL de référence", "état": "État",
    }).sort_values("RUL de référence")
    show_table(table)


# ---------------------------------------------------------------------
# JALON 1 - MLP
# ---------------------------------------------------------------------
elif page == MENU[1]:
    page_title("Durée de vie restante",
               "Le perceptron multicouche estime la RUL à partir des capteurs et de la charge.", step=0)
    model = get_model("jalon1_mlp_rul.keras")
    if model is not None:
        pump, dfp, cycle, idx, _ = pump_controls("mlp")
        try:
            shape = input_shape_of(model)
            expected = shape[-1]
            if len(shape) != 2 or expected is None or expected % N_FEATURES != 0:
                raise ValueError(f"Entrée du MLP non reconnue : {shape} (attendu : fenêtre × {N_FEATURES}).")
            steps = expected // N_FEATURES
            predictions = predict_rul(model, flat_windows(dfp, steps))
            estimate = float(predictions[idx])
            reference = min(int(dfp.rul.iloc[idx]), RUL_MAX)
            rmse = float(np.sqrt(np.mean((predictions - np.minimum(dfp.rul.values, RUL_MAX)) ** 2)))

            rul_result(estimate, reference, {"Fenêtre du modèle": f"{steps} cycles",
                                             "RMSE sur cette pompe": f"{rmse:.1f} cycles"})
            section("Évolution de la RUL sur toute la vie de la pompe")
            show_chart(trajectory_chart(dfp, predictions, cycle, estimate, "Estimation MLP"))
        except Exception as exc:
            st.error(f"Prédiction MLP impossible : {exc}")


# ---------------------------------------------------------------------
# JALON 2 - AUTO-ENCODEUR
# ---------------------------------------------------------------------
elif page == MENU[2]:
    page_title("Cycles atypiques",
               "L'auto-encodeur reconstruit les mesures ; une erreur élevée signale un cycle qui sort de l'ordinaire.",
               step=1)
    model = get_model("modele_jalon2_ae.keras")
    if model is not None:
        pump, dfp, _, _, right = pump_controls("ae", with_cycle=False)
        shape = input_shape_of(model)
        if len(shape) != 2 or shape[-1] is None or shape[-1] % N_FEATURES != 0:
            st.error(f"Entrée de l'auto-encodeur non reconnue : {shape} (attendu : fenêtre × {N_FEATURES}).")
        else:
            window = shape[-1] // N_FEATURES
            threshold = right.number_input("Seuil d'anomalie (erreur quadratique moyenne)",
                                           min_value=0.0, value=0.857, step=0.01, format="%.3f")
            try:
                windows = flat_windows(dfp, window)
                reconstruction = model.predict(windows, verbose=0, batch_size=512)
                errors = np.mean((windows - reconstruction) ** 2, axis=1)
                flagged = errors > threshold

                readouts([
                    ("Erreur maximale", f"{errors.max():.3f}", "", ""),
                    ("Erreur moyenne", f"{errors.mean():.3f}", "", ""),
                    ("Cycles au-dessus du seuil", int(flagged.sum()), f"sur {len(errors)}",
                     "crit" if flagged.any() else "ok"),
                ])

                section("Erreur de reconstruction au fil des cycles")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dfp.cycle, y=errors, mode="lines", name="Erreur de reconstruction",
                                         line=dict(color=STEEL, width=2)))
                if flagged.any():
                    fig.add_trace(go.Scatter(x=dfp.cycle[flagged], y=errors[flagged], mode="markers",
                                             name="Au-dessus du seuil",
                                             marker=dict(size=8, color=CRIT)))
                fig.add_hline(y=threshold, line_dash="dash", line_color=CRIT,
                              annotation_text="Seuil", annotation_position="top left")
                fig.update_layout(xaxis_title="Cycle", yaxis_title="Erreur quadratique moyenne")
                show_chart(chart_style(fig, 380))

                if flagged.any():
                    section("Cycles signalés")
                    flagged_df = pd.DataFrame({
                        "Cycle": dfp.cycle[flagged].values,
                        "Charge": dfp.charge[flagged].values,
                        "Erreur": errors[flagged],
                        "RUL de référence": np.minimum(dfp.rul[flagged].values, RUL_MAX),
                    })
                    show_table(flagged_df)
                else:
                    note("Aucun cycle ne dépasse le seuil pour cette pompe.")
            except Exception as exc:
                st.error(f"Analyse de l'auto-encodeur impossible : {exc}")


# ---------------------------------------------------------------------
# JALON 3 - CNN + TRANSFERT
# ---------------------------------------------------------------------
elif page == MENU[3]:
    page_title("Modes de fonctionnement et transfert",
               "Un CNN lit le spectrogramme pour classer le mode ; un second modèle estime la RUL après transfert.",
               step=2)
    tab_cnn, tab_transfer = st.tabs(["Classification du mode", "Transfert de la RUL"])

    with tab_cnn:
        cnn_file = st.selectbox("Modèle", ["modele_cnn_jalon3.keras", "modele_jalon2_cnn.keras"],
                                key="cnn_file")
        cnn = get_model(cnn_file)
        if cnn is not None:
            if "val" not in SPECTRA:
                st.warning("Aucun spectrogramme (clé spectro_val ou équivalente) dans le fichier NPZ de "
                           f"{fleet_name} : le CNN ne peut pas être alimenté.")
            else:
                pump, dfp, cycle, idx, _ = pump_controls("cnn")
                row = dfp.iloc[idx]
                try:
                    sample = np.asarray(SPECTRA["val"][int(row["_npz_index"])], dtype="float32")
                    expected = tuple(input_shape_of(cnn)[1:])
                    if any(dim is None for dim in expected):
                        raise ValueError(f"Entrée du CNN non définie : {expected}")
                    if tuple(sample.shape) != expected:
                        if sample.size != int(np.prod(expected)):
                            raise ValueError(f"Spectrogramme {sample.shape}, entrée attendue {expected}.")
                        sample = sample.reshape(expected)
                    probs = np.asarray(cnn.predict(sample[np.newaxis, ...], verbose=0)).reshape(-1)

                    modes = [f"Mode {i}" for i in range(len(probs))]
                    best = int(np.argmax(probs))
                    items = [("Mode le plus probable", modes[best], "", ""),
                             ("Confiance", f"{100 * probs[best]:.1f}", "%", "")]
                    if "mode" in dfp.columns:
                        true_mode = int(row["mode"])
                        items.insert(1, ("Mode de référence", f"Mode {true_mode}", "",
                                         "ok" if true_mode == best else "crit"))
                    readouts(items)

                    left, right = st.columns([1, 1.3], gap="large")
                    with left:
                        spec = px.imshow(sample.reshape(sample.shape[0], sample.shape[1]),
                                         color_continuous_scale="Purples", aspect="equal")
                        spec.update_coloraxes(showscale=False)
                        spec.update_xaxes(showticklabels=False)
                        spec.update_yaxes(showticklabels=False)
                        spec.update_layout(title=dict(text="Spectrogramme du cycle", font=dict(size=14)))
                        show_chart(chart_style(spec, 330))
                    with right:
                        bars = go.Figure(go.Bar(
                            x=100 * probs, y=modes, orientation="h",
                            marker_color=[STEEL if i == best else SOFT for i in range(len(probs))],
                            text=[f"{100 * p:.1f} %" for p in probs], textposition="outside",
                            cliponaxis=False,
                        ))
                        bars.update_layout(xaxis_title="Probabilité (%)", yaxis_title="",
                                           showlegend=False,
                                           title=dict(text="Probabilité de chaque mode", font=dict(size=14)))
                        bars.update_yaxes(autorange="reversed")
                        bars.update_xaxes(range=[0, 112])
                        show_chart(chart_style(bars, 330))
                except Exception as exc:
                    st.error(f"Classification impossible : {exc}")

    with tab_transfer:
        section("Comparaison des stratégies de transfert")
        results = pd.DataFrame({
            "Stratégie": ["Fine-tuning A→B", "A + statistiques B", "B from scratch", "A + statistiques A"],
            "RMSE": [17.18, 17.25, 17.57, 48.19],
            "Écart-type": [3.13, 3.00, 3.25, 1.79],
        })
        left, right = st.columns([1, 1.3], gap="large")
        with left:
            show_table(results)
        with right:
            fig = go.Figure(go.Bar(
                x=results["RMSE"], y=results["Stratégie"], orientation="h",
                error_x=dict(type="data", array=results["Écart-type"], color=MUTED),
                marker_color=[STEEL, VIOLET, TEAL, SOFT],
            ))
            fig.update_layout(xaxis_title="RMSE (cycles)", yaxis_title="", showlegend=False)
            fig.update_yaxes(autorange="reversed")
            show_chart(chart_style(fig, 300))
        note("Résultats de validation issus du projet : plus la RMSE est basse, plus l'erreur est faible.")

        transfer = get_model("modele_rul_jalon3.keras")
        if transfer is not None:
            section("Tester le modèle de transfert")
            pump, dfp, cycle, idx, _ = pump_controls("transfer")
            try:
                shape = input_shape_of(transfer)
                if len(shape) != 2 or shape[-1] != N_FEATURES:
                    raise ValueError(f"Le modèle attend {shape}, l'app fournit {N_FEATURES} variables.")
                predictions = predict_rul(transfer, dfp[FEATURE_COLS].to_numpy("float32"))
                estimate = float(predictions[idx])
                reference = min(int(dfp.rul.iloc[idx]), RUL_MAX)
                rmse = float(np.sqrt(np.mean((predictions - np.minimum(dfp.rul.values, RUL_MAX)) ** 2)))
                rul_result(estimate, reference, {"RMSE sur cette pompe": f"{rmse:.1f} cycles"})
                show_chart(trajectory_chart(dfp, predictions, cycle, estimate, "Estimation du transfert"))
            except Exception as exc:
                st.error(f"Prédiction du modèle de transfert impossible : {exc}")


# ---------------------------------------------------------------------
# JALON 4 - RNN
# ---------------------------------------------------------------------
elif page == MENU[4]:
    model = get_model("jalon4_gru_final.h5", "jalon4_gru_final.keras", "modele_jalon4_final.keras")
    kind = rnn_kind(model) if model is not None else "RNN"
    page_title("Historique de la pompe",
               f"Un réseau récurrent ({kind}) lit les derniers cycles pour estimer la RUL.", step=3)
    if model is not None:
        pump, dfp, cycle, idx, _ = pump_controls("rnn")
        try:
            shape = input_shape_of(model)
            if len(shape) != 3 or shape[1] is None or shape[2] is None:
                raise ValueError(f"Entrée du modèle récurrent non reconnue : {shape}")
            steps, features = shape[1], shape[2]
            if features != N_FEATURES:
                raise ValueError(f"Le modèle attend {features} variables par cycle, l'app en fournit {N_FEATURES}.")
            predictions = predict_rul(model, seq_windows(dfp, steps))
            estimate = float(predictions[idx])
            reference = min(int(dfp.rul.iloc[idx]), RUL_MAX)
            rmse = float(np.sqrt(np.mean((predictions - np.minimum(dfp.rul.values, RUL_MAX)) ** 2)))

            rul_result(estimate, reference, {"Fenêtre du modèle": f"{steps} cycles",
                                             "RMSE sur cette pompe": f"{rmse:.1f} cycles"})
            section("Évolution de la RUL sur toute la vie de la pompe")
            show_chart(trajectory_chart(dfp, predictions, cycle, estimate, f"Estimation {kind}"))
        except Exception as exc:
            st.error(f"Prédiction impossible : {exc}")


# ---------------------------------------------------------------------
# FICHIERS ET DIAGNOSTIC
# ---------------------------------------------------------------------
else:
    page_title("Fichiers et diagnostic",
               "Vérifie que les données et les modèles sont trouvés et se chargent avec les versions installées.")

    section("Contenu du fichier de données")
    note(f"Clés lues dans {fleet_paths[fleet_name].name}. Les noms reconnus sont de la forme "
         "pompe_val, cycle_val, charge_val, capteurs_val, rul_val, mode_val, spectro_val "
         "(les variantes val / valid / validation et l'ordre inverse sont acceptés).")
    for message in NOTES:
        note(message)
    show_table(pd.DataFrame(INVENTORY))

    section("Modèles")
    model_files = [
        ("Jalon 1, MLP de RUL", ["jalon1_mlp_rul.keras"]),
        ("Jalon 2, auto-encodeur", ["modele_jalon2_ae.keras"]),
        ("Jalon 2, CNN", ["modele_jalon2_cnn.keras"]),
        ("Jalon 3, CNN des modes", ["modele_cnn_jalon3.keras"]),
        ("Jalon 3, RUL par transfert", ["modele_rul_jalon3.keras"]),
        ("Jalon 4, réseau récurrent", ["jalon4_gru_final.h5", "jalon4_gru_final.keras",
                                       "modele_jalon4_final.keras"]),
    ]
    rows = []
    for label, names in model_files:
        path = locate(names, MODEL_DIRS)
        if path is None:
            rows.append({"Modèle": label, "Fichier": names[0], "État": "Absent", "Entrée → sortie": ""})
            continue
        model, error = load_model(str(path))
        if error:
            rows.append({"Modèle": label, "Fichier": path.name, "État": "Erreur de chargement",
                         "Entrée → sortie": error[:140]})
        else:
            rows.append({"Modèle": label, "Fichier": path.name, "État": "Chargé",
                         "Entrée → sortie": f"{input_shape_of(model)} → {tuple(model.output_shape)}"})
    show_table(pd.DataFrame(rows))
    if KERAS_VERSION < MIN_KERAS:
        st.warning("Keras est trop ancien pour ces modèles. Installez les versions de requirements.txt.")

    section("Volumes de données")
    volumes = pd.DataFrame([
        {"Partition": split, "Observations": len(frame), "Pompes": frame.pompe.nunique()}
        for split, frame in DATA.items()
    ])
    show_table(volumes)

    section("Aperçu des données")
    split = st.selectbox("Partition", list(DATA.keys()))
    preview = DATA[split].drop(columns=["_npz_index"], errors="ignore").head(150)
    show_table(preview)
    st.download_button(
        "Télécharger l'aperçu en CSV",
        data=preview.to_csv(index=False).encode("utf-8"),
        file_name=f"{fleet_name.lower().replace(' ', '_')}_{split}_apercu.csv",
        mime="text/csv",
    )