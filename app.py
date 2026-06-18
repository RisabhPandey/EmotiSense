import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import re
from collections import Counter

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotiSense · Emotion AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Mock Model for Testing UI ──────────────────────────────────────────────────
# This ensures your app doesn't crash if the .pkl files are missing.
class MockModel:
    def predict(self, X): return [5] # Defaults to Joy
    def predict_proba(self, X): return [[0.05, 0.05, 0.10, 0.05, 0.05, 0.70]]

class MockVectorizer:
    def transform(self, text): return [text]

# ─── Load Model & Vectorizer ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        logistic_model = joblib.load("logistic_model.pkl")
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return logistic_model, tfidf_vectorizer, True
    except FileNotFoundError:
        return MockModel(), MockVectorizer(), False

logistic_model, tfidf_vectorizer, model_loaded_successfully = load_model()

# ─── Constants ───────────────────────────────────────────────────────────────────
EMOTIONS = {
    0: {"label": "Sadness",  "emoji": "😢", "color": "#5B8DEF", "bg": "#1a2744"},
    1: {"label": "Anger",    "emoji": "😠", "color": "#EF5B5B", "bg": "#2e1111"},
    2: {"label": "Love",     "emoji": "❤️", "color": "#EF5BA1", "bg": "#2e1129"},
    3: {"label": "Surprise", "emoji": "😮", "color": "#EFD65B", "bg": "#2e2a11"},
    4: {"label": "Fear",     "emoji": "😰", "color": "#A35BEF", "bg": "#200e2e"},
    5: {"label": "Joy",      "emoji": "😊", "color": "#5BEF8D", "bg": "#0e2e1a"},
}

EXAMPLE_TEXTS = [
    "I can't stop smiling today — everything feels perfect!",
    "I'm so furious. Nobody listens to me and it's infuriating.",
    "I miss you so deeply. Every moment without you feels empty.",
    "She suddenly appeared at the door — I had no idea she was coming!",
    "I'm terrified of what might happen next. My hands won't stop shaking.",
    "This is the best day of my life. I'm absolutely on top of the world!",
]

# ─── Injection CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ── Base ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0d0d14;
    color: #e8e8f0;
}

/* ── Hide streamlit chrome ────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #12121e;
    border-right: 1px solid #1e1e30;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Custom header ────────────────────────────────────── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.25rem;
}
.brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin: 0;
}
.brand-sub {
    font-size: 0.9rem;
    color: #6b6b8a;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 4px;
}
.divider {
    border: none;
    border-top: 1px solid #1e1e30;
    margin: 1.25rem 0;
}

/* ── Streamlit text area overrides ───────────────────── */
textarea {
    background: #0d0d14 !important;
    color: #e8e8f0 !important;
    border: 1px solid #2a2a40 !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s !important;
}
textarea:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 3px rgba(124,106,247,0.15) !important;
}

/* ── Primary button ───────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c6af7 0%, #5b8def 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1.75rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(124,106,247,0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,106,247,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Emotion result card ──────────────────────────────── */
.result-card {
    border-radius: 16px;
    padding: 1.75rem 2rem;
    text-align: center;
    margin-bottom: 1.25rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0.08;
    background: radial-gradient(circle at 30% 30%, currentColor 0%, transparent 70%);
}
.result-emoji {
    font-size: 4rem;
    line-height: 1;
    margin-bottom: 0.5rem;
    display: block;
    animation: pop 0.4s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes pop {
    0%  { transform: scale(0.5); opacity: 0; }
    100%{ transform: scale(1);   opacity: 1; }
}
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.result-confidence {
    font-size: 0.9rem;
    color: #9999b8;
    margin-top: 0.5rem;
}

/* ── Stat cards ───────────────────────────────────────── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}
.stat-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #a78bfa;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #6b6b8a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* ── Prob bar ─────────────────────────────────────────── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.55rem;
}
.prob-label {
    width: 110px;
    font-size: 0.875rem;
    color: #c0c0d8;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: #1e1e30;
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}
.prob-pct {
    width: 42px;
    text-align: right;
    font-size: 0.82rem;
    color: #9999b8;
    flex-shrink: 0;
}

/* ── History pills ────────────────────────────────────── */
.hist-item {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.75rem;
}
.hist-text { font-size: 0.85rem; color: #b0b0cc; flex: 1; }
.hist-badge {
    font-size: 0.78rem;
    font-weight: 600;
    border-radius: 99px;
    padding: 2px 10px;
    white-space: nowrap;
}

/* ── Example chips ────────────────────────────────────── */
.chip-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.5rem; }
.chip {
    background: #1a1a2e;
    border: 1px solid #2a2a40;
    border-radius: 99px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #9090b8;
    cursor: pointer;
    transition: all 0.2s;
}
.chip:hover { background: #222238; border-color: #7c6af7; color: #c0b8f8; }

/* ── Sidebar stat ─────────────────────────────────────── */
.sb-stat { text-align: center; padding: 0.75rem; }
.sb-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #a78bfa;
}
.sb-lbl { font-size: 0.75rem; color: #6b6b8a; text-transform: uppercase; letter-spacing: 0.07em; }

/* ── Section heading ──────────────────────────────────── */
.sec-head {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b6b8a;
    margin-bottom: 0.75rem;
    font-weight: 600;
}

/* ── Selectbox & radio ────────────────────────────────── */
[data-testid="stSelectbox"], [data-testid="stRadio"] {
    color: #e8e8f0;
}

/* ── Alert override ───────────────────────────────────── */
.stAlert { background: #13131f !important; border-radius: 10px !important; }

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d14; }
::-webkit-scrollbar-thumb { background: #2a2a40; border-radius: 3px; }

/* ── Hide empty textarea label & its wrapper gap ─────── */
[data-testid="stTextArea"] label,
[data-testid="stTextArea"] > div:first-child:empty,
[data-testid="stTextArea"] [data-testid="InputInstructions"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stTextArea"] > label[data-testid="stWidgetLabel"] {
    display: none !important;
}
/* Remove top margin Streamlit adds before the widget */
[data-testid="stTextArea"] { margin-top: 0 !important; }
/* Collapse the visually-hidden label (label_visibility=hidden) */
[data-testid="stTextArea"] [data-testid="stWidgetLabel"] {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    padding: 0 !important;
    margin: -1px !important;
    overflow: hidden !important;
    clip: rect(0,0,0,0) !important;
    white-space: nowrap !important;
    border: 0 !important;
}
[data-testid="stTextArea"] > div:has(label) {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = 0
if "emotion_counts" not in st.session_state:
    st.session_state.emotion_counts = Counter()
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ─── Helper Functions ────────────────────────────────────────────────────────────
def predict_emotion(text: str):
    X = tfidf_vectorizer.transform([text])
    pred = logistic_model.predict(X)[0]
    probs = logistic_model.predict_proba(X)[0]
    return pred, probs

def word_count(text: str) -> int:
    return len(text.split())

def char_count(text: str) -> int:
    return len(text)

def avg_word_length(text: str) -> float:
    words = re.findall(r'\b\w+\b', text)
    if not words: return 0.0
    return round(sum(len(w) for w in words) / max(len(words), 1), 1)

def sentiment_intensity(probs, pred) -> str:
    p = probs[pred]
    if p >= 0.85: return "Very Strong"
    if p >= 0.65: return "Strong"
    if p >= 0.45: return "Moderate"
    return "Mild"

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:1.5rem;'>
        <div style='font-family:"Space Grotesk",sans-serif; font-size:1.3rem; font-weight:700;
                    background:linear-gradient(135deg,#a78bfa,#60a5fa); -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent; background-clip:text;'>
            🧠 EmotiSense
        </div>
        <div style='font-size:0.72rem; color:#6b6b8a; margin-top:3px; letter-spacing:0.08em; text-transform:uppercase;'>
            Emotion Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Missing model warning
    if not model_loaded_successfully:
        st.warning("⚠️ Model files not found. Running in UI Mock Mode.")

    # Session stats
    st.markdown("<div class='sec-head'>Session Stats</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='sb-stat'>
            <div class='sb-num'>{st.session_state.total_analyzed}</div>
            <div class='sb-lbl'>Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        top_e = st.session_state.emotion_counts.most_common(1)
        top_emoji = EMOTIONS[top_e[0][0]]["emoji"] if top_e else "–"
        st.markdown(f"""
        <div class='sb-stat'>
            <div class='sb-num'>{top_emoji}</div>
            <div class='sb-lbl'>Top Emotion</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Emotion legend
    st.markdown("<div class='sec-head'>Emotion Classes</div>", unsafe_allow_html=True)
    for idx, e in EMOTIONS.items():
        count = st.session_state.emotion_counts.get(idx, 0)
        st.markdown(f"""
        <div style='display:flex; align-items:center; justify-content:space-between;
                    padding:5px 0; border-bottom:1px solid #1a1a28;'>
            <span style='font-size:0.88rem; color:#c0c0d8;'>{e["emoji"]} {e["label"]}</span>
            <span style='font-size:0.78rem; color:{e["color"]}; font-weight:600;'>{count}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Action Buttons
    col_dl, col_clr = st.columns(2)
    with col_clr:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history = []
            st.session_state.total_analyzed = 0
            st.session_state.emotion_counts = Counter()
            st.rerun()
            
    with col_dl:
        if st.session_state.history:
            # Create a dataframe for download
            df = pd.DataFrame(st.session_state.history)
            df['emotion_label'] = df['pred'].apply(lambda x: EMOTIONS[x]['label'])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export", data=csv, file_name="emotisense_history.csv", mime="text/csv", use_container_width=True)
        else:
            st.button("📥 Export", disabled=True, use_container_width=True)

    # About
    st.markdown("<br><div class='sec-head'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem; color:#7070a0; line-height:1.6;'>
        EmotiSense uses a <strong style='color:#a78bfa;'>Logistic Regression</strong> model with
        <strong style='color:#60a5fa;'>TF-IDF</strong> features to detect emotions across
        6 categories in real time.
    </div>""", unsafe_allow_html=True)

# ─── Main Layout ──────────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div class='brand-header'>
    <div>
        <div class='brand-title'>EmotiSense</div>
        <div class='brand-sub'>AI-powered emotion detection · 6 emotion classes</div>
    </div>
</div>
<hr class='divider'>
""", unsafe_allow_html=True)

# Two-column layout
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    # ── Input Card ───
    st.markdown("<div class='sec-head'>Analyse Text</div>", unsafe_allow_html=True)

    user_text = st.text_area(
        label="text_input_label",
        placeholder="Type or paste any text here — a sentence, a tweet, a journal entry…",
        height=130,
        value=st.session_state.prefill,
        key="text_input",
        label_visibility="hidden"
    )

    col_btn, col_wc = st.columns([2, 1])
    with col_btn:
        predict_button = st.button("🔍 Detect Emotion", use_container_width=True)
    with col_wc:
        wc = word_count(user_text) if user_text else 0
        st.markdown(f"""
        <div style='text-align:center; padding-top:8px; font-size:0.82rem; color:#6b6b8a;'>
            {wc} word{"s" if wc != 1 else ""}
        </div>""", unsafe_allow_html=True)


    # ── Examples ─────
    st.markdown("<div class='sec-head'>Try an example</div>", unsafe_allow_html=True)
    chip_cols = st.columns(2)
    for i, ex in enumerate(EXAMPLE_TEXTS):
        with chip_cols[i % 2]:
            short = ex[:40] + "…" if len(ex) > 40 else ex
            if st.button(f"💬 {short}", key=f"ex_{i}", use_container_width=True):
                st.session_state.prefill = ex
                st.rerun()

    # ── History ──────
    if st.session_state.history:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-head'>Recent Analyses</div>", unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            e = EMOTIONS[item["pred"]]
            preview = item["text"][:55] + "…" if len(item["text"]) > 55 else item["text"]
            st.markdown(f"""
            <div class='hist-item'>
                <div class='hist-text'>{preview}</div>
                <div class='hist-badge' style='background:{e["bg"]};color:{e["color"]};border:1px solid {e["color"]}33;'>
                    {e["emoji"]} {e["label"]}
                </div>
            </div>""", unsafe_allow_html=True)

with right:
    if predict_button and user_text.strip():
        with st.spinner("Analysing emotional context…"):
            time.sleep(0.35)   # small delay so spinner is visible
            pred, probs = predict_emotion(user_text)
            st.toast('Analysis complete!', icon='✅')

        e = EMOTIONS[pred]
        confidence = probs[pred]
        intensity = sentiment_intensity(probs, pred)

        # ── Easter Egg: Joy Balloons ──
        if pred == 5 and confidence > 0.85:
            st.balloons()

        # ── Update state ──
        st.session_state.total_analyzed += 1
        st.session_state.emotion_counts[pred] += 1
        st.session_state.history.append({"text": user_text, "pred": pred, "conf": float(confidence)})
        st.session_state.prefill = ""

        # ── Result card ───
        st.markdown(f"""
        <div class='result-card' style='background:{e["bg"]};border-color:{e["color"]}44;color:{e["color"]};'>
            <span class='result-emoji'>{e["emoji"]}</span>
            <div class='result-label'>{e["label"]}</div>
            <div class='result-confidence'>{intensity} signal · {confidence*100:.1f}% confidence</div>
        </div>""", unsafe_allow_html=True)
        
        # ── Low Confidence Warning ──
        if confidence < 0.40:
             st.markdown("""
            <div style='background:#1e1a10; border:1px solid #4a3a20; border-radius:10px;
                        padding:0.75rem; color:#e0c090; font-size:0.85rem; margin-bottom:1.25rem; text-align:center;'>
                ⚠️ <b>Low Confidence:</b> The model is unsure about this text. It may contain mixed emotions or lack strong emotional keywords.
            </div>""", unsafe_allow_html=True)

        # ── Stats row ─────
        st.markdown(f"""
        <div class='stat-row'>
            <div class='stat-card'>
                <div class='stat-val'>{word_count(user_text)}</div>
                <div class='stat-lbl'>Words</div>
            </div>
            <div class='stat-card'>
                <div class='stat-val'>{char_count(user_text)}</div>
                <div class='stat-lbl'>Characters</div>
            </div>
            <div class='stat-card'>
                <div class='stat-val'>{avg_word_length(user_text)}</div>
                <div class='stat-lbl'>Avg Word Len</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Probability distribution ───
        st.markdown("<div class='sec-head'>Emotion Distribution</div>", unsafe_allow_html=True)
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        for idx, p in sorted_probs:
            em = EMOTIONS[idx]
            bar_width = int(p * 100)
            is_top = idx == pred
            opacity = "1" if is_top else "0.55"
            st.markdown(f"""
            <div class='prob-row' style='opacity:{opacity};'>
                <div class='prob-label'>{em["emoji"]} {em["label"]}</div>
                <div class='prob-bar-bg'>
                    <div class='prob-bar-fill'
                         style='width:{bar_width}%;background:{em["color"]};
                                {"box-shadow:0 0 8px "+em["color"]+"88;" if is_top else ""}'>
                    </div>
                </div>
                <div class='prob-pct'>{p*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # ── Insight blurb ──
        insight_map = {
            0: "Your text carries notes of **melancholy or grief** — often a signal of processing loss or longing.",
            1: "Strong **frustration or displeasure** detected — the language is charged with tension.",
            2: "Warm tones of **affection and tenderness** shine through your words.",
            3: "A sense of **wonder or unexpectedness** permeates the text.",
            4: "The model picks up on **anxious or fearful** undertones in your writing.",
            5: "Radiant **positivity and delight** — your words are uplifting!",
        }
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#13131f; border-left:3px solid {e["color"]};
                    border-radius:0 10px 10px 0; padding:0.9rem 1.1rem;
                    font-size:0.88rem; color:#c0c0d8; line-height:1.6;'>
            💡 {insight_map[pred]}
        </div>""", unsafe_allow_html=True)

    elif predict_button and not user_text.strip():
        st.markdown("""
        <div style='background:#1e1018; border:1px solid #4a2030; border-radius:12px;
                    padding:1.25rem; color:#e09090; font-size:0.9rem; margin-top:1rem;'>
            ⚠️ Please enter some text before running the analysis.
        </div>""", unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div style='background:#13131f; border:1px dashed #222235; border-radius:16px;
                    padding:3rem 2rem; text-align:center; margin-top:0.5rem;'>
            <div style='font-size:3rem; margin-bottom:1rem; opacity:0.4;'>🧠</div>
            <div style='font-family:"Space Grotesk",sans-serif; font-size:1.05rem;
                        color:#4a4a6a; font-weight:500;'>
                Results will appear here
            </div>
            <div style='font-size:0.82rem; color:#3a3a58; margin-top:0.4rem;'>
                Enter text on the left and click Detect Emotion
            </div>
        </div>""", unsafe_allow_html=True)