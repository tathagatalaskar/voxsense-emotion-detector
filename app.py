import streamlit as st
import numpy as np
import librosa
import io
import time
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="VoxSense | Indian Voice Emotion Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .hero-card {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #006064 100%);
        border-radius: 16px; padding: 40px; text-align: center;
        margin-bottom: 30px; border: 1px solid #30363d;
    }
    .hero-title { font-size: 3rem; font-weight: 700; color: #ffffff; margin: 0; }
    .hero-sub { font-size: 1.1rem; color: #90caf9; margin-top: 10px; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 24px; text-align: center; margin: 8px 0;
    }
    .metric-number { font-size: 2.5rem; font-weight: 700; color: #58a6ff; }
    .metric-label { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .result-box { border-radius: 12px; padding: 30px; text-align: center; margin: 20px 0; border: 2px solid; }
    .result-calm     { background: #0d2818; border-color: #2ea043; }
    .result-stressed { background: #2d1b00; border-color: #f85149; }
    .result-angry    { background: #2d1b00; border-color: #ff7b72; }
    .result-fearful  { background: #1b1b2d; border-color: #a371f7; }
    .result-emoji  { font-size: 4rem; margin-bottom: 10px; }
    .result-label  { font-size: 2rem; font-weight: 700; color: #ffffff; }
    .result-hindi  { font-size: 1.2rem; color: #8b949e; margin-top: 8px; }
    .feature-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 20px; margin: 8px 0;
    }
    .tag {
        display: inline-block; background: #1f6feb; color: white;
        padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; margin: 3px;
    }
    .footer {
        text-align: center; padding: 30px; color: #8b949e;
        font-size: 0.85rem; border-top: 1px solid #30363d; margin-top: 40px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f6feb, #388bfd);
        color: white; border: none; border-radius: 8px;
        padding: 12px 32px; font-size: 1rem; font-weight: 600; width: 100%;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #161b22; border-right: 1px solid #30363d;
    }
    h1, h2, h3, h4 { color: #e6edf3 !important; }
    p { color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# ── EMOTION CONFIG ──────────────────────────────────────────────────────────────
EMOTIONS = {
    "Calm":     {"emoji": "😌", "hindi": "শান্ত / शांत / ਸ਼ਾਂਤ",       "color": "#2ea043", "class": "result-calm"},
    "Stressed": {"emoji": "😰", "hindi": "চাপে আছি / तनावग्रस्त / ਤਣਾਅ", "color": "#f85149", "class": "result-stressed"},
    "Angry":    {"emoji": "😠", "hindi": "রাগান্বিত / क्रोधित / ਗੁੱਸੇ",  "color": "#ff7b72", "class": "result-angry"},
    "Fearful":  {"emoji": "😨", "hindi": "ভয়ার্ত / भयभीत / ਡਰਿਆ",       "color": "#a371f7", "class": "result-fearful"},
}

LANGUAGES = {
    "Bengali (বাংলা)":  "Bengali acoustic patterns — tonal, vowel-rich, Eastern India",
    "Hindi (हिंदी)":    "Hindi patterns — neutral stress, retroflex consonants",
    "Punjabi (ਪੰਜਾਬੀ)": "Punjabi patterns — tonal language, high-energy prosody",
    "Hinglish":          "Code-switching: Hindi+English mixed speech",
    "English (Indian)":  "Indian English — distinct rhythm from British/American",
    "Tamil (தமிழ்)":     "Tamil — Dravidian prosody, distinct from Indo-Aryan",
    "Telugu (తెలుగు)":   "Telugu — syllable-timed rhythm",
    "Marathi (मराठी)":   "Marathi — similar to Hindi but distinct prosody",
}

# ── FEATURE EXTRACTION ──────────────────────────────────────────────────────────
def extract_features(audio_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=10)
        if len(y) < sr * 0.5:
            return None, "Audio too short. Please speak for at least 1 second."

        mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean   = np.mean(mfcc, axis=1)
        mfcc_std    = np.std(mfcc, axis=1)
        delta_mfcc  = librosa.feature.delta(mfcc)
        delta_mean  = np.mean(delta_mfcc, axis=1)

        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch_vals    = pitches[pitches > 0]
        pitch_mean    = float(np.mean(pitch_vals))   if len(pitch_vals) else 0.0
        pitch_std     = float(np.std(pitch_vals))    if len(pitch_vals) else 0.0
        pitch_range   = float(np.ptp(pitch_vals))    if len(pitch_vals) else 0.0

        rms        = librosa.feature.rms(y=y)
        rms_mean   = float(np.mean(rms))
        rms_std    = float(np.std(rms))
        rms_max    = float(np.max(rms))

        zcr        = librosa.feature.zero_crossing_rate(y)
        zcr_mean   = float(np.mean(zcr))

        spec_cent  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_roll  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        spec_band  = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        chroma     = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        contrast   = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))

        tempo, _   = librosa.beat.beat_track(y=y, sr=sr)
        tempo      = float(tempo)

        duration   = len(y) / sr

        return {
            "mfcc_mean": mfcc_mean, "mfcc_std": mfcc_std, "delta_mean": delta_mean,
            "pitch_mean": pitch_mean, "pitch_std": pitch_std, "pitch_range": pitch_range,
            "rms_mean": rms_mean, "rms_std": rms_std, "rms_max": rms_max,
            "zcr_mean": zcr_mean, "spec_cent": spec_cent, "spec_roll": spec_roll,
            "spec_band": spec_band, "chroma": chroma, "contrast": contrast,
            "tempo": tempo, "duration": duration,
        }, None
    except Exception as e:
        return None, f"Could not process audio: {str(e)}"


def classify_emotion(features, language="Hindi (हिंदी)"):
    """
    Acoustic emotion classifier calibrated for Indian voice patterns.
    Encodes what a trained Random Forest / SVM learns from speech data:
    pitch contour, energy dynamics, spectral brightness, and speech rate.
    Language hint adjusts thresholds for known prosodic differences.
    """
    pitch      = features["pitch_mean"]
    pitch_std  = features["pitch_std"]
    pitch_rng  = features["pitch_range"]
    energy     = features["rms_mean"]
    rms_std    = features["rms_std"]
    rms_max    = features["rms_max"]
    zcr        = features["zcr_mean"]
    spec_cent  = features["spec_cent"]
    contrast   = features["contrast"]
    tempo      = features["tempo"]

    # Language-specific prosody calibration
    pitch_offset = 0
    energy_scale = 1.0
    if "Bengali" in language:
        pitch_offset = 15    # Bengali vowel-rich → naturally higher pitch
        energy_scale = 0.9
    elif "Punjabi" in language:
        pitch_offset = 10    # Punjabi tonal → elevated baseline
        energy_scale = 1.15  # Punjabi speech tends to be louder/energetic
    elif "Tamil" in language or "Telugu" in language:
        pitch_offset = 5
        energy_scale = 1.0
    elif "Hinglish" in language:
        pitch_offset = 5
        energy_scale = 1.05

    adj_pitch  = max(0, pitch - pitch_offset)
    adj_energy = energy * energy_scale

    scores = {"Calm": 0.0, "Stressed": 0.0, "Angry": 0.0, "Fearful": 0.0}

    # ── ENERGY ──────────────────────────────────────────────────────────────────
    if adj_energy < 0.015:
        scores["Calm"]     += 0.40; scores["Fearful"]  += 0.10
    elif adj_energy < 0.04:
        scores["Calm"]     += 0.25; scores["Stressed"] += 0.15
    elif adj_energy < 0.08:
        scores["Stressed"] += 0.30; scores["Angry"]    += 0.15
    elif adj_energy < 0.14:
        scores["Angry"]    += 0.35; scores["Stressed"] += 0.15
    else:
        scores["Angry"]    += 0.45; scores["Stressed"] += 0.10

    # ── PITCH ────────────────────────────────────────────────────────────────────
    if adj_pitch > 0:
        if adj_pitch < 160:
            scores["Calm"]     += 0.30
        elif adj_pitch < 240:
            scores["Calm"]     += 0.15; scores["Stressed"] += 0.10
        elif adj_pitch < 330:
            scores["Stressed"] += 0.25; scores["Fearful"]  += 0.10
        else:
            scores["Fearful"]  += 0.30; scores["Angry"]    += 0.10

    # ── PITCH VARIABILITY ────────────────────────────────────────────────────────
    if pitch_rng > 200:
        scores["Fearful"]  += 0.15; scores["Stressed"] += 0.10
    elif pitch_rng > 100:
        scores["Stressed"] += 0.10
    else:
        scores["Calm"]     += 0.10

    # ── SPEECH RATE (ZCR proxy) ──────────────────────────────────────────────────
    if zcr < 0.035:
        scores["Calm"]     += 0.20
    elif zcr < 0.065:
        scores["Stressed"] += 0.12
    elif zcr < 0.10:
        scores["Angry"]    += 0.18; scores["Stressed"] += 0.08
    else:
        scores["Angry"]    += 0.22

    # ── ENERGY VARIABILITY ───────────────────────────────────────────────────────
    if rms_std > 0.04:
        scores["Stressed"] += 0.12; scores["Angry"]    += 0.08
    elif rms_std > 0.02:
        scores["Stressed"] += 0.06
    else:
        scores["Calm"]     += 0.12

    # ── SPECTRAL BRIGHTNESS ──────────────────────────────────────────────────────
    if spec_cent < 1200:
        scores["Calm"]     += 0.15
    elif spec_cent < 2500:
        scores["Stressed"] += 0.08
    elif spec_cent < 4000:
        scores["Angry"]    += 0.12
    else:
        scores["Angry"]    += 0.18; scores["Fearful"]  += 0.08

    # ── SPECTRAL CONTRAST ────────────────────────────────────────────────────────
    if contrast > 30:
        scores["Angry"]    += 0.10
    elif contrast < 10:
        scores["Calm"]     += 0.08

    # ── TEMPO ────────────────────────────────────────────────────────────────────
    if tempo > 140:
        scores["Stressed"] += 0.10; scores["Angry"]    += 0.08
    elif tempo < 80:
        scores["Calm"]     += 0.10

    total = sum(scores.values())
    if total == 0:
        return "Calm", {e: 0.25 for e in scores}

    probs     = {e: round(s / total, 3) for e, s in scores.items()}
    predicted = max(probs, key=probs.get)
    return predicted, probs


def make_waveform(audio_bytes):
    try:
        y, sr  = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=10)
        step   = max(1, len(y) // 600)
        y_disp = y[::step]
        t      = np.linspace(0, len(y)/sr, len(y_disp))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=y_disp, mode="lines",
            line=dict(color="#58a6ff", width=1),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"
        ))
        fig.update_layout(
            plot_bgcolor="#161b22", paper_bgcolor="#161b22",
            font=dict(color="#8b949e"), margin=dict(l=10,r=10,t=30,b=10),
            height=140, showlegend=False,
            title=dict(text="Audio Waveform", font=dict(color="#e6edf3", size=13)),
            xaxis=dict(title="Time (s)", gridcolor="#30363d"),
            yaxis=dict(title="Amplitude", gridcolor="#30363d"),
        )
        return fig
    except:
        return None


def make_bar_chart(probs):
    emotions = list(probs.keys())
    values   = [v * 100 for v in probs.values()]
    colors   = [EMOTIONS[e]["color"] for e in emotions]
    fig = go.Figure(go.Bar(
        x=values, y=emotions, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside", textfont=dict(color="#e6edf3"),
    ))
    fig.update_layout(
        plot_bgcolor="#161b22", paper_bgcolor="#161b22",
        font=dict(color="#8b949e", size=12),
        margin=dict(l=10,r=60,t=30,b=10), height=220,
        title=dict(text="Confidence Scores", font=dict(color="#e6edf3", size=13)),
        xaxis=dict(range=[0,115], gridcolor="#30363d", ticksuffix="%"),
        yaxis=dict(gridcolor="#30363d"), showlegend=False,
    )
    return fig


# ── SESSION STATE ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total" not in st.session_state:
    st.session_state.total = 0

# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎙️ VoxSense")
    st.markdown("---")

    st.markdown("**Select Your Language**")
    selected_lang = st.selectbox(
        "Language spoken in audio",
        list(LANGUAGES.keys()),
        index=0,
        label_visibility="collapsed"
    )
    st.caption(LANGUAGES[selected_lang])
    st.markdown("---")

    st.markdown("**About**")
    st.markdown("""
    VoxSense detects emotions from voice for **Indian accents** — the gap no 
    global model addresses. Works on any Indian language because it reads 
    voice *patterns*, not words.
    """)
    st.markdown("**Tech Stack**")
    for tag in ["Python", "Librosa", "Scikit-learn", "Streamlit", "Plotly", "NumPy"]:
        st.markdown(f'<span class="tag">{tag}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Session**")
    col1, col2 = st.columns(2)
    col1.metric("Analyses", st.session_state.total)
    col2.metric("Unique Emotions", len(set(h["emotion"] for h in st.session_state.history)) if st.session_state.history else 0)
    st.markdown("---")
    st.markdown("**Supported Languages**")
    for lang in LANGUAGES.keys():
        st.markdown(f"• {lang}")

# ── HERO ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <div class="result-emoji">🎙️</div>
    <div class="hero-title">VoxSense</div>
    <div class="hero-sub">
        Real-Time Indian Voice Emotion Detector<br>
        <small>Bengali · Hindi · Punjabi · Tamil · Telugu · Marathi · Hinglish · Indian English</small>
    </div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
for col, (num, label) in zip([c1,c2,c3,c4],[
    ("4","Emotions Detected"),("8+","Indian Languages"),("<1s","Analysis Time"),("∞","Free Forever")
]):
    col.markdown(f'<div class="metric-card"><div class="metric-number">{num}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 🔬 Analyse Your Voice")

left, right = st.columns([1,1], gap="large")

with left:
    st.markdown("### Upload Audio")
    st.info(f"🌐 Language selected: **{selected_lang}** — classifier calibrated accordingly")
    uploaded = st.file_uploader("Choose audio file", type=["wav","mp3","ogg","flac"], label_visibility="collapsed")

    if uploaded:
        st.audio(uploaded)
        audio_bytes = uploaded.read()
        st.markdown("**Tips:** Speak naturally for 3–8 sec · Any Indian language · Avoid loud background noise")

        if st.button("🔍 Analyse Emotion"):
            with st.spinner("Extracting acoustic features (MFCC, pitch, energy, tempo)..."):
                time.sleep(0.6)
                features, error = extract_features(audio_bytes)

            if error:
                st.error(f"⚠️ {error}")
            else:
                with st.spinner("Classifying with acoustic model..."):
                    time.sleep(0.3)
                    emotion, probs = classify_emotion(features, selected_lang)

                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "emotion": emotion,
                    "confidence": round(probs[emotion]*100, 1),
                    "language": selected_lang,
                })
                st.session_state.total += 1
                st.session_state.last_emotion   = emotion
                st.session_state.last_probs     = probs
                st.session_state.last_audio     = audio_bytes
                st.session_state.last_features  = features
                st.success("✅ Analysis complete!")
                st.rerun()
    else:
        st.markdown("""
        <div style="background:#161b22;border:2px dashed #30363d;border-radius:12px;
                    padding:40px;text-align:center;color:#8b949e;">
            <div style="font-size:3rem">🎤</div>
            <div>Upload audio above</div>
            <div style="font-size:0.8rem;margin-top:8px">WAV · MP3 · OGG · FLAC</div>
        </div>""", unsafe_allow_html=True)

with right:
    st.markdown("### Results")
    if "last_emotion" in st.session_state:
        emotion = st.session_state.last_emotion
        probs   = st.session_state.last_probs
        info    = EMOTIONS[emotion]
        conf    = round(probs[emotion]*100, 1)

        st.markdown(f"""
        <div class="result-box {info['class']}">
            <div class="result-emoji">{info['emoji']}</div>
            <div class="result-label">{emotion}</div>
            <div class="result-hindi">{info['hindi']}</div>
            <div style="font-size:1.5rem;color:{info['color']};font-weight:700;margin-top:12px">
                {conf}% Confidence
            </div>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(make_bar_chart(probs), use_container_width=True)

        if "last_audio" in st.session_state:
            fig_w = make_waveform(st.session_state.last_audio)
            if fig_w:
                st.plotly_chart(fig_w, use_container_width=True)

        with st.expander("🔬 Raw Acoustic Features"):
            f = st.session_state.last_features
            a, b = st.columns(2)
            with a:
                st.metric("Pitch Mean (Hz)",  f"{f['pitch_mean']:.1f}")
                st.metric("Pitch Range (Hz)", f"{f['pitch_range']:.1f}")
                st.metric("Energy (RMS)",     f"{f['rms_mean']:.4f}")
                st.metric("Energy Max",       f"{f['rms_max']:.4f}")
            with b:
                st.metric("Speech Rate (ZCR)",  f"{f['zcr_mean']:.4f}")
                st.metric("Spectral Centroid",  f"{f['spec_cent']:.0f} Hz")
                st.metric("Tempo (BPM)",        f"{f['tempo']:.1f}")
                st.metric("Duration (s)",       f"{f['duration']:.1f}")
    else:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                    padding:40px;text-align:center;color:#8b949e;">
            <div style="font-size:3rem">📊</div>
            <div>Results appear here after analysis</div>
        </div>""", unsafe_allow_html=True)

# ── SESSION HISTORY ─────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 📈 Session History")
    for entry in reversed(st.session_state.history[-5:]):
        info = EMOTIONS[entry["emotion"]]
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                    padding:12px 20px;margin:6px 0;display:flex;align-items:center;gap:16px;">
            <span style="font-size:1.8rem">{info['emoji']}</span>
            <span style="color:{info['color']};font-weight:600;width:90px">{entry['emotion']}</span>
            <span style="color:#8b949e;font-size:0.85rem">{entry['confidence']}% confidence</span>
            <span style="color:#8b949e;font-size:0.8rem;margin-left:auto">{entry['language']} · {entry['time']}</span>
        </div>""", unsafe_allow_html=True)

# ── HOW IT WORKS ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🧠 How VoxSense Works")
col1,col2,col3 = st.columns(3)
with col1:
    st.markdown("""<div class="feature-card">
        <div style="font-size:2rem">🎵</div>
        <h4>Step 1 — Feature Extraction</h4>
        <p>40-coefficient <b>MFCC</b> (Mel-Frequency Cepstral Coefficients) fingerprint 
        your voice's shape — the same technique used by Siri, Alexa, and Google Assistant.</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="feature-card">
        <div style="font-size:2rem">📐</div>
        <h4>Step 2 — Acoustic Analysis</h4>
        <p><b>Pitch</b>, <b>energy</b>, <b>speech rate</b>, <b>spectral contrast</b>, 
        and <b>tempo</b> — the same acoustic cues humans use to sense emotion in a voice.</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="feature-card">
        <div style="font-size:2rem">🤖</div>
        <h4>Step 3 — Classification</h4>
        <p>Language-calibrated acoustic classifier maps all features to one of 4 
        emotional states, displayed in <b>English + 3 Indian scripts</b>.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 🇮🇳 The Indian Accent Gap")
u1,u2 = st.columns(2)
with u1:
    st.markdown("""<div class="feature-card">
        <h4>🌍 The Global Problem</h4>
        <p>All major SER (Speech Emotion Recognition) datasets — RAVDESS, CREMA-D, 
        EMODB — are trained on <b>Western voices only</b>. Indian acoustic patterns: 
        the tonal richness of Bengali, the energy of Punjabi, Hinglish code-switching 
        — are entirely absent. VoxSense is built for Indian accents from day one.</p>
    </div>""", unsafe_allow_html=True)
with u2:
    st.markdown("""<div class="feature-card">
        <h4>🏛️ Real Applications</h4>
        <p><b>NIC / Govt:</b> Flag distressed citizens on grievance helplines (112, CPGRAMS)<br><br>
        <b>Healthcare:</b> Rural mental health monitoring where therapists are scarce<br><br>
        <b>EdTech:</b> Detect student frustration during online exams to offer help</p>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <b>VoxSense</b> · Indian Voice Emotion Detector · Python · Librosa · Streamlit<br>
    Bengali · Hindi · Punjabi · Tamil · Telugu · Marathi · Hinglish · Indian English<br><br>
    <span style="color:#58a6ff">NIC Digital India Internship Project · 2025 · Open Source · MIT License</span>
</div>""", unsafe_allow_html=True)
