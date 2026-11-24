import json
import os
from io import BytesIO
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
from PIL import Image
from pypdf import PdfReader

# --- PAGE CONFIG (Mobile + Desktop Responsive) ---
st.set_page_config(
    page_title="MediPlain AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",
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

# --- SESSION STATE (My Reports Vault) ---
if "vault" not in st.session_state:
    st.session_state.vault = {
        "Sample: Complete Blood Count (CBC)": (
            "Hemoglobin: 10.8 g/dL (Ref: 13.0-17.0) | "
            "WBC: 7,400 /uL (Ref: 4,000-11,000) | "
            "Platelets: 245,000 /uL (Ref: 150,000-450,000) | "
            "MCV: 74 fL (Ref: 80-100)"
        )
    }
if "active_report" not in st.session_state:
    st.session_state.active_report = "Sample: Complete Blood Count (CBC)"
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def text_to_audio(text: str, lang_code: str) -> BytesIO:
    tts = gTTS(text=text[:600], lang=lang_code)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp


def analyze_report(client, report_text: str, language: str):
    prompt = f"""
    You are a compassionate clinical communication assistant. Explain every medical biomarker
    in this lab report clearly in {language} for a patient with no medical background.
    Return ONLY valid JSON with this exact schema:
    {{
      "summary": "2-3 sentence plain-language overview in {language}",
      "biomarkers": [
        {{
          "name": "Test name + abbreviation",
          "value": "Patient value with unit",
          "reference_range": "Normal range",
          "status": "Normal | Borderline | High | Low",
          "what_it_is": "Simple 1-sentence explanation in {language}",
          "why_it_matters": "Why the body needs it in {language}",
          "what_result_means": "What this specific value means in {language}"
        }}
      ],
      "doctor_questions": ["3 practical questions to ask a doctor in {language}"]
    }}

    Report text:
    {report_text}
    """
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text)


# --- SIDEBAR: API KEY, LANGUAGE & MY REPORTS VAULT ---
with st.sidebar:
    st.title("🩺 MediPlain AI")
    api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    selected_lang = st.selectbox("Preferred Language", list(LANGUAGES.keys()))
    lang_code = LANGUAGES[selected_lang]
    auto_voice = st.toggle("🔊 Auto-Voice Readout", value=True)

    st.divider()
    st.subheader("📁 My Reports Vault")
    uploaded = st.file_uploader("Upload Report (PDF, Image, TXT)", type=["pdf", "png", "jpg", "jpeg", "txt"])

    if uploaded is not None:
        if uploaded.name not in st.session_state.vault:
            if uploaded.type == "application/pdf":
                reader = PdfReader(uploaded)
                extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
            elif uploaded.type.startswith("image/"):
                if not api_key:
                    st.warning("Enter your Gemini API Key first to read image reports.")
                    extracted = ""
                else:
                    client = genai.Client(api_key=api_key)
                    img = Image.open(uploaded)
                    ocr_res = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=["Extract all text and lab values verbatim from this medical report.", img],
                    )
                    extracted = ocr_res.text
            else:
                extracted = uploaded.read().decode("utf-8", errors="ignore")

            if extracted.strip():
                st.session_state.vault[uploaded.name] = extracted
                st.session_state.active_report = uploaded.name
                st.success(f"Saved '{uploaded.name}' to Vault!")

    report_names = list(st.session_state.vault.keys())
    st.session_state.active_report = st.radio(
        "Select Saved Report",
        report_names,
        index=report_names.index(st.session_state.active_report),
    )

# --- MAIN AREA ---
st.header(f"📄 {st.session_state.active_report}")
st.caption("⚠️ Educational health-literacy tool only — not a medical diagnosis. Always consult a qualified doctor.")

if not api_key:
    st.info("👈 Paste your free **Gemini API Key** from Google AI Studio in the left sidebar (tap `>` on mobile) to start.")
    st.stop()

client = genai.Client(api_key=api_key)
current_text = st.session_state.vault[st.session_state.active_report]
cache_key = f"{st.session_state.active_report}_{selected_lang}"

tab1, tab2, tab3 = st.tabs(["🔬 Term-by-Term Breakdown", "💬 Voice & Text Chatbot", "📃 Raw Report"])

with tab1:
    if cache_key not in st.session_state.analysis_cache:
        with st.spinner(f"Translating and explaining every biomarker in {selected_lang}..."):
            st.session_state.analysis_cache[cache_key] = analyze_report(client, current_text, selected_lang)

    data = st.session_state.analysis_cache[cache_key]
    st.subheader("Plain-Language Summary")
    st.write(data["summary"])
    if auto_voice:
        st.audio(text_to_audio(data["summary"], lang_code), format="audio/mp3")

    st.subheader("Every Term Explained Neatly")
    for item in data["biomarkers"]:
        status = item.get("status", "Normal")
        badge = "🟢" if status == "Normal" else ("🟡" if status == "Borderline" else "🔴")
        with st.expander(f"{badge} {item['name']} — {item['value']} ({status})", expanded=True):
            st.markdown(f"**Reference Range:** `{item['reference_range']}`")
            st.markdown(f"**What It Is:** {item['what_it_is']}")
            st.markdown(f"**Why It Matters:** {item['why_it_matters']}")
            st.markdown(f"**What Your Result Means:** {item['what_result_means']}")
            if st.button(f"🔊 Listen ({item['name']})", key=f"tts_{cache_key}_{item['name']}"):
                spoken = f"{item['name']}. {item['what_it_is']} {item['what_result_means']}"
                st.audio(text_to_audio(spoken, lang_code), format="audio/mp3")

    st.subheader("🩺 Questions to Ask Your Doctor")
    for q in data["doctor_questions"]:
        st.markdown(f"- {q}")

with tab2:
    st.subheader(f"Ask About Your Report ({selected_lang})")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Native Streamlit Voice Input (works on both Laptop Mic & Mobile Phone Mic!)
    voice_audio = st.audio_input("🎙️ Tap to ask by voice (or type below)")
    user_text = st.chat_input(f"Type your question in {selected_lang}...")

    user_prompt_parts = []
    display_user_msg = None

    if voice_audio is not None:
        audio_bytes = voice_audio.read()
        user_prompt_parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))
        user_prompt_parts.append(
            f"Listen to the patient's voice question and reply in {selected_lang} based strictly on this medical report:\n{current_text}"
        )
        display_user_msg = "🎙️ [Voice Question Sent]"
    elif user_text:
        user_prompt_parts.append(
            f"Answer the patient's question in {selected_lang} based strictly on this medical report:\n{current_text}\n\nQuestion: {user_text}"
        )
        display_user_msg = user_text

    if display_user_msg:
        st.session_state.chat_history.append({"role": "user", "content": display_user_msg})
        with st.chat_message("user"):
            st.write(display_user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=user_prompt_parts,
                ).text
                st.write(reply)
                if auto_voice:
                    st.audio(text_to_audio(reply, lang_code), format="audio/mp3")
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

with tab3:
    st.text_area("Extracted Report Text", current_text, height=300)
