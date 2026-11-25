import json
import os
import re
import time
from io import BytesIO
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
from PIL import Image
from pypdf import PdfReader

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MediPlain AI | Medical Report Explainer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CLEAN UI STYLING ---
st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    .hero-box {
        background: linear-gradient(135deg, #0f766e 0%, #0284c7 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.15);
    }
    .hero-box h1 { margin: 0; font-size: 1.8rem; font-weight: 700; color: white; }
    .hero-box p { margin: 0.4rem 0 0 0; font-size: 0.95rem; opacity: 0.92; }
    .summary-card {
        background: white;
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        border-left: 5px solid #0284c7;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .bio-card {
        background: white;
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        margin-bottom: 0.9rem;
    }
    .bio-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 0.6rem;
    }
    .bio-title { font-size: 1.1rem; font-weight: 700; color: #0f172a; }
    .bio-val { font-size: 0.95rem; font-weight: 600; color: #334155; }
    .badge-normal { background: #dcfce7; color: #166534; padding: 3px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .badge-borderline { background: #fef9c3; color: #854d0e; padding: 3px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .badge-high { background: #fee2e2; color: #991b1b; padding: 3px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .doctor-box {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

LANGUAGES = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Bengali (বাংলা)": "bn",
    "Marathi (मराठी)": "mr",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
    "Arabic (العربية)": "ar",
}

SAMPLE_REPORTS = {
    "Sample 1: Complete Blood Count (CBC)": (
        "Hemoglobin: 10.8 g/dL (Ref: 13.0-17.0)\n"
        "WBC Count: 7,400 /uL (Ref: 4,000-11,000)\n"
        "Platelets: 245,000 /uL (Ref: 150,000-450,000)\n"
        "MCV (Mean Corpuscular Volume): 74 fL (Ref: 80-100)"
    ),
    "Sample 2: Lipid Profile & HbA1c": (
        "HbA1c: 6.1 % (Ref: 4.0-5.6 Normal, 5.7-6.4 Prediabetes)\n"
        "Total Cholesterol: 218 mg/dL (Ref: <200)\n"
        "LDL Cholesterol: 142 mg/dL (Ref: <100)\n"
        "HDL Cholesterol: 44 mg/dL (Ref: >40)\n"
        "Triglycerides: 165 mg/dL (Ref: <150)"
    ),
}

# Primary + fallback models so each model's daily free-tier quota is utilized automatically
MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
]


def call_gemini_resilient(client, contents, config=None) -> str:
    """Tries primary and backup models, skipping immediately on 429 quota errors."""
    last_err = None
    for model_name in MODEL_CANDIDATES:
        for attempt in range(2):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                if res and res.text:
                    return res.text
            except Exception as e:
                last_err = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    break  # Daily quota full for this model; jump to next model immediately
                time.sleep(1.2)
    raise RuntimeError(f"All Gemini models temporarily busy. Please try again in a few seconds. ({last_err})")


# --- SESSION STATE ---
if "vault" not in st.session_state:
    st.session_state.vault = dict(SAMPLE_REPORTS)
if "active_report" not in st.session_state:
    st.session_state.active_report = list(SAMPLE_REPORTS.keys())[0]
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_lang" not in st.session_state:
    st.session_state.last_lang = "English"


def text_to_audio(text: str, lang_code: str):
    """Safely converts text to MP3 bytes without crashing on language switch or long text."""
    try:
        clean_text = re.sub(r"[#*`_]", "", text).strip()
        if not clean_text:
            return None
        tts = gTTS(text=clean_text[:500], lang=lang_code, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception:
        return None


def extract_json_safely(raw_text: str) -> dict:
    """Strips markdown fences if returned and parses JSON safely."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def analyze_report(client, report_text: str, language: str) -> dict:
    prompt = f"""
    You are a compassionate clinical communication assistant. Explain every medical biomarker
    in this lab report clearly in {language} for a patient with no medical background.
    Keep medical abbreviations (like HbA1c, MCV, LDL, TSH, T3, T4) recognizable, and write all explanations in {language}.

    Return ONLY valid JSON matching this exact structure:
    {{
      "summary": "2-3 sentence plain-language overview in {language}",
      "biomarkers": [
        {{
          "name": "Test name and abbreviation",
          "value": "Patient value with unit",
          "reference_range": "Normal range",
          "status": "Normal",
          "what_it_is": "Simple 1-sentence explanation in {language}",
          "why_it_matters": "Why the body needs it in {language}",
          "what_result_means": "What this specific result means in {language}"
        }}
      ],
      "doctor_questions": [
        "Question 1 in {language}",
        "Question 2 in {language}",
        "Question 3 in {language}"
      ]
    }}
    Note: "status" must be strictly one of: "Normal", "Borderline", "High", or "Low".

    Medical Report:
    {report_text}
    """
    raw = call_gemini_resilient(
        client,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return extract_json_safely(raw)


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🩺 MediPlain AI")
    st.caption("Multilingual Medical Report Vault & Voice Companion")

    default_secret = ""
    try:
        default_secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    raw_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", default_secret),
        placeholder="Paste AIza... key",
    )
    api_key = raw_key.strip().strip('"').strip("'")

    selected_lang = st.selectbox("🌐 Select Language", list(LANGUAGES.keys()))
    lang_code = LANGUAGES[selected_lang]

    if selected_lang != st.session_state.last_lang:
        st.session_state.chat_history = []
        st.session_state.last_lang = selected_lang

    auto_voice = st.toggle("🔊 Auto-Voice Readout", value=True)

    st.divider()
    st.markdown("### 📁 My Reports Vault")
    uploaded = st.file_uploader(
        "Upload New Report (PDF, Image, TXT)",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
    )

    if uploaded is not None and uploaded.name not in st.session_state.vault:
        extracted = ""
        file_bytes = uploaded.read()
        if uploaded.type == "application/pdf":
            reader = PdfReader(BytesIO(file_bytes))
            extracted = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if not extracted and api_key:
                try:
                    ocr_client = genai.Client(api_key=api_key)
                    extracted = call_gemini_resilient(
                        ocr_client,
                        contents=[
                            types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                            "Extract all medical lab test names, values, units, and reference ranges verbatim.",
                        ],
                    ).strip()
                except Exception as e:
                    st.error(f"Scanned PDF read error: {e}")
        elif uploaded.type.startswith("image/"):
            if not api_key:
                st.warning("Enter your Gemini API Key above first to read image reports.")
            else:
                try:
                    ocr_client = genai.Client(api_key=api_key)
                    # Compress/normalize image to JPEG bytes for fast, reliable OCR
                    img = Image.open(BytesIO(file_bytes)).convert("RGB")
                    img.thumbnail((1600, 1600))
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    extracted = call_gemini_resilient(
                        ocr_client,
                        contents=[
                            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                            "Extract all medical lab test names, values, units, and reference ranges verbatim.",
                        ],
                    ).strip()
                except Exception as e:
                    st.error(f"Image read error (click Retry or re-upload): {e}")
        else:
            extracted = file_bytes.decode("utf-8", errors="ignore").strip()

        if extracted:
            st.session_state.vault[uploaded.name] = extracted
            st.session_state.active_report = uploaded.name
            st.success(f"Saved '{uploaded.name}' to Vault!")

    report_names = list(st.session_state.vault.keys())
    selected_report = st.radio(
        "Saved Reports",
        report_names,
        index=report_names.index(st.session_state.active_report),
    )
    if selected_report != st.session_state.active_report:
        st.session_state.active_report = selected_report
        st.session_state.chat_history = []
        st.rerun()

# --- HERO HEADER ---
st.markdown(
    f"""
    <div class="hero-box">
        <h1>🩺 MediPlain AI — Patient Report Explainer</h1>
        <p>Active Report: <b>{st.session_state.active_report}</b> &nbsp;|&nbsp; Language: <b>{selected_lang}</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("⚠️ Educational health-literacy assistant only — not a medical diagnosis. Always consult a qualified doctor.")

if not api_key:
    st.info("👈 Paste your free **Gemini API Key** (`AIza...`) from [Google AI Studio](https://aistudio.google.com/api-keys) in the left sidebar to begin.")
    st.stop()

client = genai.Client(api_key=api_key)
current_text = st.session_state.vault[st.session_state.active_report]
cache_key = f"{st.session_state.active_report}__{selected_lang}"

tab1, tab2, tab3 = st.tabs([
    "🔬 Term-by-Term Breakdown",
    "💬 Voice & Text Chatbot",
    "📃 Original Report Text",
])

# --- TAB 1: NEAT TERM-BY-TERM EXPLAINER ---
with tab1:
    if cache_key not in st.session_state.analysis_cache:
        with st.spinner(f"Analyzing and translating every term into {selected_lang}..."):
            try:
                st.session_state.analysis_cache[cache_key] = analyze_report(
                    client, current_text, selected_lang
                )
            except Exception as e:
                st.error(f"Temporary AI service delay in {selected_lang}: {e}")
                if st.button("🔄 Retry Now"):
                    st.rerun()
                st.stop()

    data = st.session_state.analysis_cache[cache_key]

    st.markdown(
        f"""
        <div class="summary-card">
            <h4 style="margin:0 0 8px 0; color:#0284c7;">📋 Plain-Language Overview ({selected_lang})</h4>
            <p style="margin:0; font-size:1.02rem; color:#1e293b; line-height:1.6;">{data.get("summary", "")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if auto_voice:
        summary_audio = text_to_audio(data.get("summary", ""), lang_code)
        if summary_audio:
            st.audio(summary_audio, format="audio/mp3")

    st.subheader("🧪 Every Biomarker Explained Neatly")

    for idx, item in enumerate(data.get("biomarkers", [])):
        status = item.get("status", "Normal")
        badge_cls = (
            "badge-normal"
            if status.lower() == "normal"
            else ("badge-borderline" if status.lower() == "borderline" else "badge-high")
        )

        st.markdown(
            f"""
            <div class="bio-card">
                <div class="bio-header">
                    <span class="bio-title">{item.get("name", "Biomarker")}</span>
                    <div>
                        <span class="bio-val">{item.get("value", "")}</span>
                        &nbsp;<span style="color:#64748b; font-size:0.85rem;">(Ref: {item.get("reference_range", "N/A")})</span>
                        &nbsp;<span class="{badge_cls}">{status}</span>
                    </div>
                </div>
                <p style="margin:6px 0;"><b>🔹 What It Is:</b> {item.get("what_it_is", "")}</p>
                <p style="margin:6px 0;"><b>🔹 Why It Matters:</b> {item.get("why_it_matters", "")}</p>
                <p style="margin:6px 0;"><b>🔹 What Your Result Means:</b> {item.get("what_result_means", "")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(f"🔊 Listen to {item.get('name', 'Term')}", key=f"btn_{cache_key}_{idx}"):
            spoken = f"{item.get('name', '')}. {item.get('what_it_is', '')} {item.get('what_result_means', '')}"
            term_audio = text_to_audio(spoken, lang_code)
            if term_audio:
                st.audio(term_audio, format="audio/mp3")
            else:
                st.warning("Audio readout unavailable for this text snippet.")

    questions = data.get("doctor_questions", [])
    if questions:
        q_html = "".join(f"<li style='margin-bottom:6px;'>{q}</li>" for q in questions)
        st.markdown(
            f"""
            <div class="doctor-box">
                <h4 style="margin:0 0 10px 0; color:#0f766e;">🩺 Smart Questions to Ask Your Doctor</h4>
                <ul style="margin:0; padding-left:20px; color:#1e293b;">{q_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- TAB 2: VOICE & TEXT CHATBOT ---
with tab2:
    st.subheader(f"💬 Ask Questions About This Report ({selected_lang})")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    col_mic, col_clear = st.columns([4, 1])
    with col_mic:
        voice_audio = st.audio_input(
            "🎙️ Tap microphone to ask by voice",
            key=f"mic_{cache_key}_{len(st.session_state.chat_history)}",
        )
    with col_clear:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    user_text = st.chat_input(f"Type your question in {selected_lang}...")

    user_prompt_parts = []
    display_user_msg = None

    if voice_audio is not None:
        audio_bytes = voice_audio.read()
        user_prompt_parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))
        user_prompt_parts.append(
            f"Listen to the patient's voice question and answer clearly in {selected_lang} based strictly on this medical report:\n{current_text}"
        )
        display_user_msg = "🎙️ [Voice Question]"
    elif user_text:
        user_prompt_parts.append(
            f"Answer the patient's question clearly in {selected_lang} based strictly on this medical report:\n{current_text}\n\nQuestion: {user_text}"
        )
        display_user_msg = user_text

    if display_user_msg:
        st.session_state.chat_history.append({"role": "user", "content": display_user_msg})
        with st.chat_message("user"):
            st.write(display_user_msg)

        with st.chat_message("assistant"):
            with st.spinner(f"Preparing reply in {selected_lang}..."):
                try:
                    reply = call_gemini_resilient(client, contents=user_prompt_parts)
                    st.write(reply)
                    if auto_voice:
                        reply_audio = text_to_audio(reply, lang_code)
                        if reply_audio:
                            st.audio(reply_audio, format="audio/mp3")
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Chat Error: {e}")

# --- TAB 3: RAW REPORT ---
with tab3:
    st.subheader("Extracted Report Text")
    st.text_area("Raw Content", current_text, height=320, label_visibility="collapsed")
