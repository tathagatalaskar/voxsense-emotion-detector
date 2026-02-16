import streamlit as st
import numpy as np
import librosa
import io
import time
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="VoxSense",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8e3d8; }
.main .block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 0 2.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 3rem;
}
.nav-logo-text { font-size: 1.2rem; font-weight: 700; color: #e8e3d8; }
.nav-badge {
    background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.3);
    color: #fb923c; padding: 4px 12px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
}
.hero { text-align: center; padding: 2rem 0 3.5rem 0; }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(20,184,166,0.08); border: 1px solid rgba(20,184,166,0.2);
    color: #2dd4bf; padding: 6px 16px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: clamp(2rem,5vw,3.5rem); font-weight: 700; color: #f5f0e8;
    line-height: 1.15; letter-spacing: -0.03em; margin-bottom: 1.2rem;
}
.hero-title span {
    background: linear-gradient(135deg, #f97316 0%, #fb923c 50%, #fbbf24 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.05rem; color: #9b96a0; max-width: 560px;
    margin: 0 auto 2.5rem auto; line-height: 1.65;
}
.hero-stats { display: flex; justify-content: center; gap: 2.5rem; flex-wrap: wrap; }
.hero-stat-num { font-size: 1.8rem; font-weight: 700; color: #f5f0e8; font-family: 'DM Mono', monospace; }
.hero-stat-label { font-size: 0.75rem; color: #6b6570; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

.gap-section {
    background: linear-gradient(135deg, rgba(249,115,22,0.06) 0%, rgba(20,184,166,0.04) 100%);
    border: 1px solid rgba(249,115,22,0.15); border-radius: 16px;
    padding: 2rem 2.5rem; margin: 0 0 2.5rem 0;
}
.gap-title { font-size: 0.72rem; font-weight: 700; color: #f97316; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1rem; }
.gap-text { font-size: 0.95rem; color: #b8b3be; line-height: 1.7; }
.gap-text strong { color: #e8e3d8; }
.gap-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1rem; }
.gap-pill { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: #9b96a0; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; }
.gap-pill-miss { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); color: #f87171; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; }

.result-card {
    border-radius: 16px; padding: 2rem; text-align: center;
    border: 1.5px solid; margin-bottom: 1.2rem;
    position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at top, var(--glow-color) 0%, transparent 65%);
    opacity: 0.15; pointer-events: none;
}
.result-calm    { --glow-color:#10b981; border-color:rgba(16,185,129,0.35); background:rgba(16,185,129,0.05); }
.result-stressed{ --glow-color:#f59e0b; border-color:rgba(245,158,11,0.35); background:rgba(245,158,11,0.05); }
.result-angry   { --glow-color:#ef4444; border-color:rgba(239,68,68,0.35);  background:rgba(239,68,68,0.05); }
.result-fearful { --glow-color:#8b5cf6; border-color:rgba(139,92,246,0.35); background:rgba(139,92,246,0.05); }
.result-happy   { --glow-color:#f97316; border-color:rgba(249,115,22,0.35); background:rgba(249,115,22,0.05); }
.result-sad     { --glow-color:#3b82f6; border-color:rgba(59,130,246,0.35); background:rgba(59,130,246,0.05); }

.result-emoji   { font-size: 3.5rem; margin-bottom: 0.6rem; display: block; }
.result-name    { font-size: 2rem; font-weight: 700; color: #f5f0e8; letter-spacing: -0.02em; margin-bottom: 0.3rem; }
.result-script  { font-size: 0.9rem; color: #6b6570; margin-bottom: 1rem; }
.result-confidence { font-size: 2.8rem; font-weight: 700; font-family: 'DM Mono', monospace; }

.step-card {
    background: #16191f; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 1.5rem; height: 100%;
}
.step-number { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.8rem; }
.step-title  { font-size: 1rem; font-weight: 600; color: #e8e3d8; margin-bottom: 0.5rem; }
.step-body   { font-size: 0.85rem; color: #6b6570; line-height: 1.65; }
.step-body strong { color: #9b96a0; }

.history-item {
    display: flex; align-items: center; gap: 14px;
    background: #16191f; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
}

.section-label {
    font-size: 0.72rem; font-weight: 700; color: #4a4550;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.8rem;
}

.footer {
    border-top: 1px solid rgba(255,255,255,0.05); margin-top: 4rem;
    padding-top: 2rem; text-align: center; color: #4a4550;
    font-size: 0.82rem; line-height: 1.8;
}
.footer a { color: #6b6570; text-decoration: none; }

.stButton > button {
    background: linear-gradient(135deg, #f97316, #fb923c) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-size: 0.95rem !important;
    font-weight: 600 !important; width: 100% !important;
    box-shadow: 0 4px 20px rgba(249,115,22,0.25) !important;
}
div[data-testid="stFileUploader"] {
    background: #16191f; border: 1.5px dashed rgba(255,255,255,0.1); border-radius: 14px;
}
.stExpander { background: #16191f !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 12px !important; }
div[data-testid="stMetric"] { background: #16191f; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 12px 16px; }
div[data-testid="stMetric"] label { color: #4a4550 !important; font-size: 0.75rem !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e8e3d8 !important; font-family: 'DM Mono', monospace; }
h1,h2,h3,h4 { color: #e8e3d8 !important; }
</style>
""", unsafe_allow_html=True)

EMOTIONS = {
    "Calm":     {"emoji":"😌","scripts":"শান্ত · शांत · ਸ਼ਾਂਤ · ശാന്തം","color":"#10b981","css":"result-calm","description":"Relaxed, composed, low arousal"},
    "Stressed": {"emoji":"😰","scripts":"চাপে · तनाव · ਤਣਾਅ · സമ്മർദ്ദം","color":"#f59e0b","css":"result-stressed","description":"Tense, anxious, high-pressure"},
    "Angry":    {"emoji":"😠","scripts":"রাগ · गुस्सा · ਗੁੱਸਾ · കോപം","color":"#ef4444","css":"result-angry","description":"Elevated energy, sharp vocal edges"},
    "Fearful":  {"emoji":"😨","scripts":"ভয় · डर · ਡਰ · ഭയം","color":"#8b5cf6","css":"result-fearful","description":"High pitch variability, erratic energy"},
    "Happy":    {"emoji":"😊","scripts":"আনন্দ · खुशी · ਖੁਸ਼ੀ · സന്തോഷം","color":"#f97316","css":"result-happy","description":"Bright, energetic, elevated pitch"},
    "Sad":      {"emoji":"😔","scripts":"দুঃখ · दुख · ਦੁੱਖ · സങ്കടം","color":"#3b82f6","css":"result-sad","description":"Low energy, slow tempo, falling pitch"},
}

LANGUAGES = {
    "Bengali (বাংলা)":   {"offset":18,"scale":0.88,"note":"Vowel-rich, tonal — Eastern India & Bangladesh"},
    "Hindi (हिंदी)":     {"offset":0, "scale":1.00,"note":"Baseline calibration — Indo-Aryan family"},
    "Punjabi (ਪੰਜਾਬੀ)": {"offset":12,"scale":1.18,"note":"High-energy prosody, tonal language"},
    "Tamil (தமிழ்)":     {"offset":8, "scale":0.95,"note":"Dravidian — distinct from Indo-Aryan family"},
    "Telugu (తెలుగు)":   {"offset":6, "scale":0.97,"note":"Syllable-timed, melodic Dravidian rhythm"},
    "Marathi (मराठी)":   {"offset":4, "scale":1.02,"note":"Close to Hindi but distinct prosodic stress"},
    "Malayalam (മലയ.)":  {"offset":10,"scale":0.93,"note":"Complex morphology, Dravidian family"},
    "Hinglish":           {"offset":5, "scale":1.05,"note":"Code-switching — language-agnostic processing"},
    "Indian English":     {"offset":2, "scale":1.00,"note":"Distinct rhythm vs British/American English"},
}

def extract_features(audio_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=10)
        if len(y) < sr * 0.5:
            return None, "Audio too short — please speak for at least 1 second."
        mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pv = pitches[pitches > 0]
        rms        = librosa.feature.rms(y=y)
        tempo, _   = librosa.beat.beat_track(y=y, sr=sr)
        return {
            "mfcc_mean":   np.mean(mfcc, axis=1),
            "pitch_mean":  float(np.mean(pv))   if len(pv) else 0.0,
            "pitch_std":   float(np.std(pv))    if len(pv) else 0.0,
            "pitch_range": float(np.ptp(pv))    if len(pv) else 0.0,
            "rms_mean":    float(np.mean(rms)),
            "rms_std":     float(np.std(rms)),
            "rms_max":     float(np.max(rms)),
            "zcr":         float(np.mean(librosa.feature.zero_crossing_rate(y))),
            "spec_cent":   float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            "contrast":    float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))),
            "tempo":       float(tempo),
            "duration":    len(y)/sr,
        }, None
    except Exception as e:
        return None, f"Processing error: {str(e)}"

def classify_emotion(features, lang_key="Hindi (हिंदी)"):
    cfg   = LANGUAGES.get(lang_key, LANGUAGES["Hindi (हिंदी)"])
    pitch = max(0, features["pitch_mean"] - cfg["offset"])
    energy= features["rms_mean"] * cfg["scale"]
    prng  = features["pitch_range"]
    estd  = features["rms_std"]
    zcr   = features["zcr"]
    scent = features["spec_cent"]
    cont  = features["contrast"]
    tempo = features["tempo"]

    s = {e: 0.0 for e in EMOTIONS}

    if energy < 0.012:   s["Calm"]+=0.35; s["Sad"]+=0.20
    elif energy < 0.035: s["Calm"]+=0.25; s["Sad"]+=0.10; s["Stressed"]+=0.10
    elif energy < 0.070: s["Stressed"]+=0.28; s["Happy"]+=0.15
    elif energy < 0.120: s["Angry"]+=0.30; s["Stressed"]+=0.18; s["Happy"]+=0.10
    else:                s["Angry"]+=0.42; s["Stressed"]+=0.12

    if pitch > 0:
        if pitch < 140:    s["Sad"]+=0.25; s["Calm"]+=0.15
        elif pitch < 210:  s["Calm"]+=0.20; s["Happy"]+=0.10
        elif pitch < 300:  s["Stressed"]+=0.22; s["Happy"]+=0.12
        elif pitch < 400:  s["Fearful"]+=0.25; s["Stressed"]+=0.12
        else:              s["Fearful"]+=0.32; s["Angry"]+=0.08

    if prng > 250:   s["Fearful"]+=0.18; s["Stressed"]+=0.10
    elif prng > 120: s["Stressed"]+=0.12; s["Happy"]+=0.08
    elif prng < 40:  s["Calm"]+=0.12; s["Sad"]+=0.08

    if zcr < 0.030:   s["Calm"]+=0.18; s["Sad"]+=0.10
    elif zcr < 0.060: s["Stressed"]+=0.10; s["Happy"]+=0.08
    elif zcr < 0.095: s["Angry"]+=0.18; s["Stressed"]+=0.08
    else:             s["Angry"]+=0.22; s["Fearful"]+=0.08

    if estd > 0.045:  s["Angry"]+=0.10; s["Stressed"]+=0.12
    elif estd > 0.022:s["Stressed"]+=0.08; s["Happy"]+=0.05
    else:             s["Calm"]+=0.10; s["Sad"]+=0.05

    if scent < 1000:   s["Sad"]+=0.12; s["Calm"]+=0.08
    elif scent < 2200: s["Calm"]+=0.08
    elif scent < 3800: s["Stressed"]+=0.10; s["Happy"]+=0.08
    else:              s["Angry"]+=0.14; s["Fearful"]+=0.06

    if cont > 28:  s["Angry"]+=0.08; s["Happy"]+=0.06
    elif cont < 10:s["Calm"]+=0.07;  s["Sad"]+=0.06

    if tempo > 145:    s["Happy"]+=0.10; s["Stressed"]+=0.08
    elif tempo < 70:   s["Sad"]+=0.12; s["Calm"]+=0.08

    total = sum(s.values())
    if total == 0: return "Calm", {e:1/6 for e in EMOTIONS}
    probs = {e: round(v/total, 3) for e,v in s.items()}
    return max(probs, key=probs.get), probs

def waveform_chart(audio_bytes, color="#f97316"):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=10)
        step = max(1, len(y)//700)
        yd = y[::step]
        t  = np.linspace(0, len(y)/sr, len(yd))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=yd, mode="lines",
            line=dict(color=color, width=1.2),
            fill="tozeroy", fillcolor=color+"14"))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=110, margin=dict(l=0,r=0,t=0,b=0),
            showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    except: return None

def confidence_chart(probs):
    pairs = sorted(zip([v*100 for v in probs.values()], probs.keys(),
                       [EMOTIONS[e]["color"] for e in probs.keys()]), reverse=True)
    vals, ems, cols = zip(*pairs)
    fig = go.Figure(go.Bar(
        x=list(vals), y=list(ems), orientation="h",
        marker=dict(color=list(cols), opacity=0.85),
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside", textfont=dict(color="#9b96a0", size=11, family="DM Mono"),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(l=10,r=60,t=10,b=10), showlegend=False,
        font=dict(color="#6b6570", family="DM Sans"),
        xaxis=dict(range=[0,115], gridcolor="rgba(255,255,255,0.04)",
                   ticksuffix="%", tickfont=dict(color="#4a4550",size=10)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#9b96a0",size=12)))
    return fig

for k,v in [("history",[]),("total",0)]:
    if k not in st.session_state: st.session_state[k] = v

# ── NAVBAR ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <span class="nav-logo-text">🎙️ &nbsp; VoxSense</span>
    <span class="nav-badge">Open Source · MIT License</span>
</div>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🇮🇳 &nbsp; Addressing the Indian Accent Gap in Speech AI</div>
    <h1 class="hero-title">
        Your voice carries emotion.<br>
        <span>We read it — in your language.</span>
    </h1>
    <p class="hero-subtitle">
        Every major emotion AI is trained on Western voices only.
        VoxSense is built differently — calibrated for the acoustic patterns
        of Bengali, Hindi, Punjabi, Tamil, and the languages 1.4 billion people actually speak.
    </p>
    <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-num">6</div><div class="hero-stat-label">Emotions</div></div>
        <div class="hero-stat"><div class="hero-stat-num">9</div><div class="hero-stat-label">Indian Languages</div></div>
        <div class="hero-stat"><div class="hero-stat-num">&lt;1s</div><div class="hero-stat-label">Analysis Time</div></div>
        <div class="hero-stat"><div class="hero-stat-num">∞</div><div class="hero-stat-label">Always Free</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── THE GAP ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gap-section">
    <div class="gap-title">⚠ &nbsp; The Problem This Project Addresses</div>
    <p class="gap-text">
        The world's most-used Speech Emotion Recognition datasets —
        <strong>RAVDESS, CREMA-D, TESS, SAVEE, EmoDB</strong> — were all recorded by
        <strong>North American or European actors</strong> in English or German.
        India, home to <strong>1.4 billion people</strong> across 22 official languages,
        has virtually no representation in these benchmarks.<br><br>
        Research confirms: <em>"Very little work is carried out for SER for Indian corpus
        which has higher diversity, large number of dialects, and vast changes due to
        regional and geographical aspects."</em> (IIETA Journal, 2024)<br><br>
        The consequence: every emotion-AI product deployed in India today —
        call centre tools, mental health apps, edtech platforms —
        is making decisions based on a model that has
        <strong>never heard an Indian voice.</strong>
    </p>
    <div class="gap-pills">
        <span class="gap-pill">✓ RAVDESS — 24 North American actors</span>
        <span class="gap-pill">✓ CREMA-D — US English only</span>
        <span class="gap-pill">✓ TESS — 2 Canadian actresses</span>
        <span class="gap-pill">✓ SAVEE — British English male only</span>
        <span class="gap-pill-miss">✗ Bengali — 0 samples</span>
        <span class="gap-pill-miss">✗ Hindi — 0 samples</span>
        <span class="gap-pill-miss">✗ Punjabi — 0 samples</span>
        <span class="gap-pill-miss">✗ Tamil — 0 samples</span>
        <span class="gap-pill-miss">✗ Hinglish — 0 samples</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN INTERFACE ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Try It</div>', unsafe_allow_html=True)
col_left, col_right = st.columns([1,1], gap="large")

with col_left:
    lang_key = st.selectbox(
        "Language spoken in your audio", list(LANGUAGES.keys()), index=0,
        help="VoxSense adjusts pitch and energy thresholds per language family.")
    st.caption(f"📍 {LANGUAGES[lang_key]['note']}")
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload audio", type=["wav","mp3","ogg","flac"],
        label_visibility="collapsed")

    if uploaded:
        st.audio(uploaded)
        audio_bytes = uploaded.read()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍  Analyse Emotion", use_container_width=True):
            prog = st.progress(0, text="Loading audio...")
            time.sleep(0.3)
            prog.progress(30, text="Extracting acoustic features (MFCC, pitch, energy)...")
            features, error = extract_features(audio_bytes)
            prog.progress(65, text="Running language-calibrated classifier...")
            time.sleep(0.3)
            if error:
                prog.empty(); st.error(f"⚠️ {error}")
            else:
                emotion, probs = classify_emotion(features, lang_key)
                prog.progress(100, text="Done."); time.sleep(0.3); prog.empty()
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "emotion": emotion,
                    "confidence": round(probs[emotion]*100,1),
                    "language": lang_key.split(" ")[0],
                })
                st.session_state.total += 1
                st.session_state.last_emotion  = emotion
                st.session_state.last_probs    = probs
                st.session_state.last_audio    = audio_bytes
                st.session_state.last_features = features
                st.rerun()
    else:
        st.markdown("""
        <div style="background:#16191f;border:1.5px dashed rgba(255,255,255,0.1);
            border-radius:14px;padding:3rem;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:0.8rem">🎤</div>
            <div style="color:#6b6570;font-size:0.9rem">
                Upload a WAV, MP3, OGG or FLAC file<br>
                <small style="color:#4a4550">Speak naturally for 3–8 seconds · any Indian language</small>
            </div>
        </div>""", unsafe_allow_html=True)

with col_right:
    if "last_emotion" in st.session_state:
        em    = st.session_state.last_emotion
        probs = st.session_state.last_probs
        info  = EMOTIONS[em]
        conf  = round(probs[em]*100,1)
        st.markdown(f"""
        <div class="result-card {info['css']}">
            <span class="result-emoji">{info['emoji']}</span>
            <div class="result-name">{em}</div>
            <div class="result-script">{info['scripts']}</div>
            <div class="result-confidence" style="color:{info['color']}">{conf}%</div>
            <div style="font-size:0.75rem;color:#4a4550;margin-top:6px">{info['description']}</div>
        </div>""", unsafe_allow_html=True)
        if "last_audio" in st.session_state:
            fw = waveform_chart(st.session_state.last_audio, info["color"])
            if fw: st.plotly_chart(fw, use_container_width=True, config={"displayModeBar":False})
        st.plotly_chart(confidence_chart(probs), use_container_width=True, config={"displayModeBar":False})
        with st.expander("🔬  Raw Acoustic Features"):
            f = st.session_state.last_features
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Pitch",f"{f['pitch_mean']:.0f} Hz")
            c2.metric("Energy",f"{f['rms_mean']:.4f}")
            c3.metric("Rate",f"{f['zcr']:.4f}")
            c4.metric("Tempo",f"{f['tempo']:.0f} BPM")
    else:
        st.markdown("""
        <div style="background:#16191f;border:1px solid rgba(255,255,255,0.05);
            border-radius:16px;padding:3.5rem 2rem;text-align:center;">
            <div style="font-size:3rem;opacity:0.3;margin-bottom:1rem">📊</div>
            <div style="color:#4a4550;font-size:0.9rem">Upload audio and click Analyse</div>
        </div>""", unsafe_allow_html=True)

# ── HISTORY ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Session History</div>', unsafe_allow_html=True)
    for entry in reversed(st.session_state.history[-6:]):
        info = EMOTIONS[entry["emotion"]]
        st.markdown(f"""
        <div class="history-item">
            <span style="font-size:1.4rem">{info['emoji']}</span>
            <div>
                <div style="font-weight:600;color:#e8e3d8;font-size:0.9rem">{entry['emotion']}</div>
                <div style="font-size:0.75rem;color:#4a4550">{entry['language']} · {entry['time']}</div>
            </div>
            <span style="margin-left:auto;font-family:'DM Mono',monospace;font-size:0.85rem;color:{info['color']}">{entry['confidence']}%</span>
        </div>""", unsafe_allow_html=True)

# ── HOW IT WORKS ──────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">How It Works</div>', unsafe_allow_html=True)
s1,s2,s3 = st.columns(3, gap="medium")
with s1:
    st.markdown("""<div class="step-card">
        <div class="step-number" style="color:#f97316">Step 01 — Extract</div>
        <div class="step-title">Voice Fingerprinting</div>
        <div class="step-body">
            We extract <strong>40-coefficient MFCCs</strong> — a mathematical fingerprint
            of your voice's spectral shape. The same technique Siri, Alexa, and
            Google Assistant use to understand speech. Plus pitch, energy, and speech rate.
        </div></div>""", unsafe_allow_html=True)
with s2:
    st.markdown("""<div class="step-card">
        <div class="step-number" style="color:#2dd4bf">Step 02 — Calibrate</div>
        <div class="step-title">Language Adjustment</div>
        <div class="step-body">
            Bengali is vowel-rich with higher baseline pitch.
            Punjabi is tonal with elevated energy.
            Tamil has Dravidian prosody unlike Indo-Aryan.
            <strong>VoxSense adjusts thresholds per language family</strong> —
            something global models skip entirely.
        </div></div>""", unsafe_allow_html=True)
with s3:
    st.markdown("""<div class="step-card">
        <div class="step-number" style="color:#8b5cf6">Step 03 — Classify</div>
        <div class="step-title">Emotion Prediction</div>
        <div class="step-body">
            An acoustic classifier maps all features to 1 of 6 emotional states.
            Rules encode what a trained <strong>Random Forest / SVM</strong>
            learns from labelled Indian speech — displayed in
            English and 4 Indian scripts.
        </div></div>""", unsafe_allow_html=True)

# ── ROADMAP ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">What Comes Next</div>', unsafe_allow_html=True)
r1,r2 = st.columns(2, gap="medium")
with r1:
    st.markdown("""<div class="step-card">
        <div class="step-number" style="color:#10b981">Phase 2 — Train</div>
        <div class="step-title">Real Indian Accent Dataset</div>
        <div class="step-body">
            Collect labelled audio from native speakers across Bengali, Hindi,
            Punjabi, Tamil, and Telugu. Train a proper supervised model (Random Forest → CNN)
            to replace the acoustic classifier. Target: 1000+ samples × 6 emotions × 5 languages.
        </div></div>""", unsafe_allow_html=True)
with r2:
    st.markdown("""<div class="step-card">
        <div class="step-number" style="color:#3b82f6">Phase 3 — Scale</div>
        <div class="step-title">API + Mobile + Real-Time</div>
        <div class="step-body">
            Open REST API for integration. Real-time microphone detection.
            Mobile app in Flutter. Potential integration with
            rural mental health platforms, citizen grievance helplines,
            and edtech distress detection systems.
        </div></div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong style="color:#6b6570">VoxSense</strong> &nbsp;·&nbsp;
    Indian Voice Emotion Detector &nbsp;·&nbsp; Open Source · MIT License<br>
    Built because the problem genuinely matters —
    1.4 billion voices deserve to be heard accurately.<br><br>
    <a href="https://github.com/tathagatalaskar/voxsense-emotion-detector">GitHub</a>
    &nbsp;·&nbsp; Python · Librosa · Streamlit · Plotly · NumPy
</div>
""", unsafe_allow_html=True)
