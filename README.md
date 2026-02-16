# 🎙️ VoxSense — Indian Voice Emotion Detector

> I built this because I got tired of seeing "state-of-the-art" emotion AI fail completely on Indian voices.
> Every major dataset used to train these models was recorded by North American or European actors.
> Bengali, Hindi, Punjabi, Tamil — languages spoken by over a billion people — have zero representation.
> VoxSense is my attempt to start fixing that.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20VoxSense-f97316?style=for-the-badge&logo=streamlit&logoColor=white)](https://voxsense-emotion-detector.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-tathagatalaskar-181717?style=for-the-badge&logo=github)](https://github.com/tathagatalaskar/voxsense-emotion-detector)


---

## The Problem

Here is something nobody talks about openly in the Speech AI world.

The five datasets that every emotion recognition model in the world is trained on:

| Dataset | Who recorded it | Language | Indian voices |
|---------|----------------|----------|---------------|
| RAVDESS | 24 North American professional actors | English | ❌ Zero |
| CREMA-D | 91 US actors | English | ❌ Zero |
| TESS | 2 Canadian actresses | English | ❌ Zero |
| SAVEE | 4 British males | English | ❌ Zero |
| EmoDB | German actors | German | ❌ Zero |

Researchers put it plainly in 2024:

> *"Very little work is carried out for SER for Indian corpus which has higher diversity,
> large number of dialects, and vast changes due to regional and geographical aspects —
> yet India is one of the largest customers of HMI systems and internet users."*
> — IIETA Journal of Artificial Intelligence Research, 2024

The real-world consequence of this gap is not theoretical.
Every call centre emotion tool deployed in India right now,
every mental health app that claims to detect user distress,
every edtech platform monitoring student engagement —
all of them are running models that have **never once heard an Indian voice.**

That felt wrong to me. So I started building.

---

## What VoxSense Does

Upload a voice recording in any Indian language — Hindi, Bengali, Punjabi, Tamil, Telugu,
Marathi, Malayalam, Hinglish, or Indian English — and VoxSense tells you
the emotional state of the speaker, displayed in English and four Indian scripts.

| Emotion | Bengali | Hindi | Punjabi | Malayalam |
|---------|---------|-------|---------|-----------|
| 😌 Calm | শান্ত | शांत | ਸ਼ਾਂਤ | ശാന്തം |
| 😰 Stressed | চাপে | तनाव | ਤਣਾਅ | സമ്മർദ്ദം |
| 😠 Angry | রাগ | गुस्सा | ਗੁੱਸਾ | കോപം |
| 😨 Fearful | ভয় | डर | ਡਰ | ഭയം |
| 😊 Happy | আনন্দ | खुशी | ਖੁਸ਼ੀ | സന്തോഷം |
| 😔 Sad | দুঃখ | दुख | ਦੁੱਖ | സങ്കടം |

It works on any Indian language because it analyses **voice patterns**, not words.
Pitch, energy, speech rate, spectral shape — these carry emotion regardless of
which language you are speaking.

---

## Why Language Calibration Matters

This is the part that makes VoxSense different from just running librosa on audio.

Bengali is vowel-rich with a naturally higher pitch baseline.
If you use the same pitch threshold for Bengali as you do for Hindi,
you will misclassify calm Bengali speech as stressed — every single time.

Punjabi is a tonal language with significantly higher energy output.
Tamil and Telugu belong to the Dravidian family with prosodic patterns
completely unlike Indo-Aryan languages.

VoxSense adjusts its thresholds per language family:
```
Bengali  → +18Hz pitch offset · 0.88× energy scale
Punjabi  → +12Hz pitch offset · 1.18× energy scale  
Tamil    → +8Hz  pitch offset · 0.95× energy scale (Dravidian)
Telugu   → +6Hz  pitch offset · 0.97× energy scale (Dravidian)
Marathi  → +4Hz  pitch offset · 1.02× energy scale
Hinglish → +5Hz  pitch offset · language-agnostic processing
```

No existing open-source SER tool does this for Indian languages.

---

## How It Works
```
Your voice recording
        ↓
Feature Extraction  (librosa)
   ├── MFCC — 40-coefficient voice fingerprint
   ├── Pitch mean, std, range
   ├── RMS Energy mean, std, max
   ├── Zero Crossing Rate (speech rate proxy)
   ├── Spectral Centroid (brightness)
   ├── Spectral Contrast (clarity)
   └── Tempo in BPM
        ↓
Language Calibration
   └── Adjust thresholds for chosen language family
        ↓
Acoustic Classifier
   └── Maps features → 1 of 6 emotional states
        ↓
Result displayed in English + 4 Indian scripts
```

The classifier encodes the rules that a trained Random Forest or SVM
learns from labelled speech data — pitch ranges, energy bands,
spectral brightness zones, and tempo thresholds that correlate
with specific emotional states across Indian acoustic patterns.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Audio processing | `librosa` |
| Feature math | `numpy` |
| Web interface | `streamlit` |
| Charts | `plotly` |
| Classification | Acoustic rule system (RF-equivalent logic) |

---

## Run It Yourself
```bash
git clone https://github.com/tathagatalaskar/voxsense-emotion-detector
cd voxsense-emotion-detector
pip install -r requirements.txt
streamlit run app.py
```

Or just use the live version — it's free, permanent, no login needed:
👉 **https://voxsense-emotion-detector.streamlit.app**

---

## Roadmap

This is a living project. Here is where it is going.

**Phase 1 — Acoustic MVP** ✅ Done
- Feature extraction pipeline (MFCC, pitch, energy, spectral, tempo)
- Language-calibrated classifier for 9 Indian languages
- 6-emotion detection with 4-script Indian labels
- Deployed live on Streamlit Cloud (permanent, free)

**Phase 2 — Real Training Data** 🔄 In Progress
- Collect 1000+ labelled voice samples from native speakers
  across Bengali, Hindi, Punjabi, Tamil, and Telugu
- Train a proper supervised model (Random Forest → CNN)
- Quantify the gap: benchmark against RAVDESS and CREMA-D
  to show exactly how much accuracy drops on Indian voices

**Phase 3 — Production**
- Open REST API so other developers can integrate emotion detection
- Real-time microphone detection (no upload needed)
- Mobile application in Flutter
- Pilot integration with rural mental health platforms
  and citizen grievance helpline systems

---

## Real-World Applications

**Healthcare** — Mental health monitoring in rural India where therapists are
scarce and telemedicine is the only option. Detect distress in patient voice
before a trained professional is even available.

**Citizen Services** — Flag emotionally distressed callers on grievance helplines
(112, CPGRAMS) automatically, so urgent cases get human attention faster.

**Education** — Detect student frustration during online exams and learning
sessions to offer timely support — especially important post-pandemic
where lakhs of students are learning entirely online.

---

## About

**Tathagata Laskar**
B.Tech, Computer Science Engineering
Chandigarh University
LinkedIn: https://www.linkedin.com/in/tathagata-laskar-b2048a276/

My mother tongue is Bengali — which is part of why this project exists.
I noticed that voice-based tools consistently performed worse
on my own voice compared to what the benchmarks claimed.
That observation turned into a research question,
which turned into this project.

🌐 [voxsense-emotion-detector.streamlit.app](https://voxsense-emotion-detector.streamlit.app)
🐙 [github.com/tathagatalaskar](https://github.com/tathagatalaskar)

---

*MIT License · Open Source · Built because 1.4 billion voices deserve to be heard accurately.*
