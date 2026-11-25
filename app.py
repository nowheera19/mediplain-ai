import json
import os
import re
import time
from io import BytesIO
import joblib
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
from PIL import Image
from pypdf import PdfReader
from train_model import build_and_save_model

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MediPlain AI | Universal Medical Document Explainer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- LOAD / AUTO-TRAIN LOCAL ML MODEL ---
MODEL_FILE = "clinical_risk_model.pkl"


@st.cache_resource
def load_trained_ml_model():
    if not os.path.exists(MODEL_FILE):
        return build_and_save_model(MODEL_FILE)
    return joblib.load(MODEL_FILE)


ml_artifact = load_trained_ml_model()

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
    .doc-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .ml-card {
        background: #f8fafc;
        border: 1.5px solid #cbd5e1;
        border-left: 6px solid #7c3aed;
        padding: 1rem 1.4rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }
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
    .badge-info { background: #e0f2fe; color: #075985; padding: 3px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
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
    "Sample 1: Complete Blood Count (Lab)": (
        "Hemoglobin: 10.8 g/dL (Ref: 13.0-17.0)\n"
        "WBC Count: 7,400 /uL (Ref: 4,000-11,000)\n"
        "Platelets: 245,000 /uL (Ref: 150,000-450,000)\n"
        "MCV (Mean Corpuscular Volume): 74 fL (Ref: 80-100)"
    ),
    "Sample 2: Pregnancy Ultrasound Report": (
        "OBSTETRIC ULTRASOUND (2nd Trimester)\n"
        "Gestational Age by LMP: 20 weeks 2 days\n"
        "Fetal Heart Rate (FHR): 148 bpm (Normal: 110-160 bpm)\n"
        "Placenta: Posterior, Grade I maturity, away from internal os\n"
        "Amniotic Fluid Index (AFI): 13.5 cm (Normal: 8-18 cm)\n"
        "Estimated Fetal Weight (EFW): 340 grams"
    ),
    "Sample 3: Chest X-Ray Radiology Report": (
        "CHEST X-RAY (PA VIEW)\n"
        "Lungs: Mild hyperinflation with minimal peribronchial thickening in bilateral lower zones. No focal consolidation or pleural effusion.\n"
        "Heart: Cardiothoracic ratio within normal limits.\n"
        "Bony Thorax & Diaphragm: Intact, costophrenic angles clear.\n"
        "Impression: Mild lower airway inflammation / bronchitis; correlate clinically."
    ),
}

# Pre-cached English analyses so opening the app & switching samples consumes ZERO API quota
PRELOADED_SAMPLE_CACHE = {
    "Sample 1: Complete Blood Count (Lab)__English": {
        "document_type": "Complete Blood Count (CBC) Lab Report",
        "summary": "Your white blood cells and platelets are healthy and normal, while your Hemoglobin and MCV (red blood cell size) are slightly below the normal range, which commonly points toward mild iron-deficiency anemia.",
        "findings": [
            {
                "name": "Hemoglobin",
                "value": "10.8 g/dL",
                "reference_range": "13.0-17.0",
                "status": "Low",
                "what_it_is": "A protein in red blood cells that carries oxygen from your lungs to the rest of your body.",
                "why_it_matters": "Adequate hemoglobin keeps your energy levels steady and prevents fatigue or shortness of breath.",
                "what_result_means": "At 10.8 g/dL, your hemoglobin is mildly below the reference range, which can explain tiredness or weakness.",
            },
            {
                "name": "WBC Count",
                "value": "7,400 /uL",
                "reference_range": "4,000-11,000",
                "status": "Normal",
                "what_it_is": "White blood cells are the immune cells that defend your body against infections.",
                "why_it_matters": "A balanced WBC count shows your immune system is functioning normally without signs of acute infection.",
                "what_result_means": "Your count of 7,400 /uL sits comfortably right in the middle of the healthy range.",
            },
            {
                "name": "Platelets",
                "value": "245,000 /uL",
                "reference_range": "150,000-450,000",
                "status": "Normal",
                "what_it_is": "Tiny blood cell fragments that help your blood clot properly when you get a cut or bruise.",
                "why_it_matters": "Healthy platelet levels prevent excessive bleeding and support normal wound healing.",
                "what_result_means": "Your platelet count of 245,000 /uL is completely normal.",
            },
            {
                "name": "MCV (Mean Corpuscular Volume)",
                "value": "74 fL",
                "reference_range": "80-100",
                "status": "Low",
                "what_it_is": "A measurement of the average size of your red blood cells.",
                "why_it_matters": "Red blood cell size helps doctors identify the specific cause of low hemoglobin.",
                "what_result_means": "An MCV of 74 fL means your red blood cells are slightly smaller than average (microcytic), often seen with low iron.",
            },
        ],
        "doctor_questions": [
            "Could my low Hemoglobin and MCV be caused by iron deficiency, and should I get an iron/ferritin blood test?",
            "Do you recommend dietary changes or an iron supplement based on these numbers?",
            "When should I repeat my Complete Blood Count to check if my levels are improving?",
        ],
    },
    "Sample 2: Pregnancy Ultrasound Report__English": {
        "document_type": "2nd Trimester Pregnancy Ultrasound Report",
        "summary": "This 20-week pregnancy ultrasound shows healthy fetal development. Your baby's heart rate, amniotic fluid level, placental position, and estimated weight are all within normal, reassuring limits.",
        "findings": [
            {
                "name": "Fetal Heart Rate (FHR)",
                "value": "148 bpm",
                "reference_range": "110-160 bpm",
                "status": "Normal",
                "what_it_is": "The number of times your baby's heart beats per minute during the scan.",
                "why_it_matters": "A steady heart rate confirms healthy blood flow and oxygen supply to the baby.",
                "what_result_means": "148 beats per minute is right in the healthy target range.",
            },
            {
                "name": "Placenta Position",
                "value": "Posterior, Grade I, away from internal os",
                "reference_range": "N/A",
                "status": "Normal",
                "what_it_is": "The organ that nourishes your baby in the womb, located along the back wall (posterior) and clear of the cervix.",
                "why_it_matters": "Being away from the internal os means the placenta is not blocking the birth canal.",
                "what_result_means": "This is a completely normal, safe placental position for 20 weeks.",
            },
            {
                "name": "Amniotic Fluid Index (AFI)",
                "value": "13.5 cm",
                "reference_range": "8-18 cm",
                "status": "Normal",
                "what_it_is": "A measurement of the protective fluid surrounding your baby inside the amniotic sac.",
                "why_it_matters": "Proper fluid volume cushions the baby and supports healthy lung and limb movement.",
                "what_result_means": "13.5 cm indicates a healthy, balanced amount of amniotic fluid.",
            },
            {
                "name": "Estimated Fetal Weight (EFW)",
                "value": "340 grams",
                "reference_range": "N/A",
                "status": "Normal",
                "what_it_is": "An ultrasound calculation of your baby's current weight based on body measurements.",
                "why_it_matters": "Tracking weight confirms steady growth matching your gestational age of 20 weeks and 2 days.",
                "what_result_means": "340 grams is right on track for a healthy 20-week pregnancy.",
            },
        ],
        "doctor_questions": [
            "Are all anatomical measurements from my 20-week anomaly scan normal?",
            "When is my next routine ultrasound or glucose screening scheduled?",
            "Are there any specific prenatal vitamins or nutrition tips I should focus on in the second half of pregnancy?",
        ],
    },
    "Sample 3: Chest X-Ray Radiology Report__English": {
        "document_type": "Chest X-Ray Radiology Report",
        "summary": "Your Chest X-Ray shows a normal heart size and clear ribcage, with mild lower airway inflammation (consistent with mild bronchitis). There are no signs of pneumonia or fluid buildup.",
        "findings": [
            {
                "name": "Lungs (Lower Zones)",
                "value": "Mild hyperinflation & minimal peribronchial thickening",
                "reference_range": "N/A",
                "status": "Borderline",
                "what_it_is": "Mild irritation and slight air trapping in the smaller breathing tubes at the bottom of the lungs.",
                "why_it_matters": "This typically happens when airways are irritated by a recent cold, cough, allergies, or mild bronchitis.",
                "what_result_means": "There is no pneumonia (focal consolidation) or fluid around the lungs (pleural effusion), only mild airway irritation.",
            },
            {
                "name": "Heart (Cardiothoracic Ratio)",
                "value": "Within normal limits",
                "reference_range": "N/A",
                "status": "Normal",
                "what_it_is": "A comparison of the width of your heart to the width of your chest cavity.",
                "why_it_matters": "A normal ratio indicates your heart is not enlarged.",
                "what_result_means": "Your heart shadow appears completely normal in size and shape.",
            },
            {
                "name": "Bony Thorax & Diaphragm",
                "value": "Intact, costophrenic angles clear",
                "reference_range": "N/A",
                "status": "Normal",
                "what_it_is": "The ribs, spine, and breathing muscle below the lungs, along with the lower corners of the lung space.",
                "why_it_matters": "Clear angles confirm there is no fluid collection or structural injury.",
                "what_result_means": "Your chest bones and lower lung margins look healthy and normal.",
            },
        ],
        "doctor_questions": [
            "Do my symptoms match mild bronchitis, and will it clear up with rest or do I need an inhaler or medication?",
            "How long should I expect a cough or airway irritation to last before feeling back to normal?",
            "What warning signs (such as fever or shortness of breath) mean I should come back sooner?",
        ],
    },
}

# Expanded pool of Gemini models (each model has its own separate daily Free Tier quota bucket)
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite",
]


def predict_ml_risk(report_text: str):
    """Extracts numeric blood parameters via regex and runs the locally trained Random Forest classifier."""
    patterns = {
        "hemoglobin": r"(?:hemoglobin|hb)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        "mcv": r"(?:mcv|mean corpuscular volume)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        "wbc": r"(?:wbc|white blood cell)[^\d]*(\d[\d,]*(?:\.\d+)?)",
        "platelets": r"(?:platelets|plt)[^\d]*(\d[\d,]*(?:\.\d+)?)",
        "hba1c": r"(?:hba1c|a1c)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        "ldl": r"(?:ldl)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
    }
    defaults = {
        "hemoglobin": 14.0,
        "mcv": 90.0,
        "wbc": 7000.0,
        "platelets": 250000.0,
        "hba1c": 5.1,
        "ldl": 90.0,
    }

    extracted_vals = {}
    matched_count = 0
    lower_txt = report_text.lower()

    for key, pat in patterns.items():
        m = re.search(pat, lower_txt)
        if m:
            val = float(m.group(1).replace(",", ""))
            extracted_vals[key] = val
            matched_count += 1
        else:
            extracted_vals[key] = defaults[key]

    if matched_count == 0:
        return None

    X_input = pd.DataFrame([extracted_vals])[ml_artifact["features"]]
    clf = ml_artifact["model"]
    pred_class = int(clf.predict(X_input)[0])
    probs = clf.predict_proba(X_input)[0]
    confidence = round(float(max(probs)) * 100, 1)

    tier_map = {
        0: ("Normal Range", "#166534", "#dcfce7"),
        1: ("Borderline / Monitor", "#854d0e", "#fef9c3"),
        2: ("High Priority / Consult Doctor", "#991b1b", "#fee2e2"),
    }
    label, fg, bg = tier_map[pred_class]
    return {
        "label": label,
        "fg": fg,
        "bg": bg,
        "confidence": confidence,
        "matched_count": matched_count,
        "accuracy": ml_artifact["accuracy"],
    }


def build_local_fallback_analysis(report_text: str) -> dict:
    """Graceful offline parser if all Gemini API quotas are exhausted for the day."""
    lines = [ln.strip() for ln in report_text.splitlines() if ln.strip()]
    findings = []
    for ln in lines[:8]:
        parts = ln.split(":", 1)
        name = parts[0].strip()
        val = parts[1].strip() if len(parts) > 1 else "Recorded"
        findings.append({
            "name": name,
            "value": val,
            "reference_range": "N/A",
            "status": "Informational",
            "what_it_is": f"Documented clinical entry for {name} extracted from your report.",
            "why_it_matters": "Helps your doctor evaluate this specific measurement or observation.",
            "what_result_means": f"Recorded value/note: {val}. Discuss with your doctor for personalized clinical context.",
        })
    return {
        "document_type": "Medical Document (Offline Parser Mode)",
        "summary": "Your document was parsed locally while the cloud AI service is cooling down from daily free-tier rate limits. Review the extracted entries and ML risk screening below.",
        "findings": findings,
        "doctor_questions": [
            "Can you walk me through which values or findings on this report need follow-up?",
            "Are any additional tests or lifestyle adjustments recommended based on these results?",
            "When should I schedule my next checkup?",
        ],
    }


def get_hidden_api_key() -> str:
    """Silently loads API key from Streamlit Secrets or environment variables."""
    secret_val = ""
    try:
        secret_val = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    raw = os.getenv("GEMINI_API_KEY", secret_val)
    return str(raw).strip().strip('"').strip("'")


def call_gemini_resilient(client, contents, config=None) -> str:
    """Tries primary and backup models, skipping immediately on 429 quota errors."""
    last_err = None
    for model_name in MODEL_CANDIDATES:
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
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str:
                continue
            time.sleep(0.8)
    raise RuntimeError(f"All Gemini free-tier models reached daily quota ({last_err})")


# --- SESSION STATE ---
if "vault" not in st.session_state:
    st.session_state.vault = dict(SAMPLE_REPORTS)
if "active_report" not in st.session_state:
    st.session_state.active_report = list(SAMPLE_REPORTS.keys())[0]
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = dict(PRELOADED_SAMPLE_CACHE)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_lang" not in st.session_state:
    st.session_state.last_lang = "English"


def text_to_audio(text: str, lang_code: str):
    """Safely converts text to MP3 bytes."""
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


def analyze_medical_document(client, report_text: str, language: str) -> dict:
    prompt = f"""
    You are a compassionate clinical communication assistant. Automatically detect what type of medical
    document this is (e.g., Blood/Lab Report, X-Ray / Radiology Scan, Pregnancy / Ultrasound Report,
    Prescription, Discharge Summary, or General Medical Document).

    Explain every key finding, biomarker, measurement, or medication clearly in {language} for a patient
    with no medical background. Keep medical abbreviations (like HbA1c, MCV, LDL, FHR, AFI, PA View)
    recognizable, and write all explanations in {language}.

    Return ONLY valid JSON matching this exact structure:
    {{
      "document_type": "Detected document category in {language} (e.g., Pregnancy Ultrasound Report, Chest X-Ray, Blood Lab Report)",
      "summary": "2-3 sentence plain-language overview of the overall report in {language}",
      "findings": [
        {{
          "name": "Test name, scan region, measurement, or medicine name",
          "value": "Measured value, observation, or dosage",
          "reference_range": "Normal range or expected baseline (use 'N/A' for X-ray/scan observations)",
          "status": "Normal",
          "what_it_is": "Simple 1-sentence explanation in {language}",
          "why_it_matters": "Why this organ, measurement, or medicine matters in {language}",
          "what_result_means": "What this specific finding means for the patient in {language}"
        }}
      ],
      "doctor_questions": [
        "Question 1 in {language}",
        "Question 2 in {language}",
        "Question 3 in {language}"
      ]
    }}
    Note: "status" must be strictly one of: "Normal", "Borderline", "High", "Low", or "Informational".

    Medical Document Content:
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


# --- LOAD HIDDEN API KEY ---
api_key = get_hidden_api_key()
if not api_key:
    st.error("⚠️ Service configuration missing: Please configure `GEMINI_API_KEY` in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# --- SIDEBAR (NO API KEY SHOWN TO USERS) ---
with st.sidebar:
    st.markdown("## 🩺 MediPlain AI")
    st.caption("Hybrid ML + Multimodal Medical Explainer")

    selected_lang = st.selectbox("🌐 Select Language", list(LANGUAGES.keys()))
    lang_code = LANGUAGES[selected_lang]

    if selected_lang != st.session_state.last_lang:
        st.session_state.chat_history = []
        st.session_state.last_lang = selected_lang

    auto_voice = st.toggle("🔊 Auto-Voice Readout", value=True)

    st.divider()
    st.markdown("### 📁 My Medical Vault")
    uploaded = st.file_uploader(
        "Upload Lab Report, X-Ray, Pregnancy Scan, or Prescription (PDF, Image, TXT)",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
    )

    if uploaded is not None and uploaded.name not in st.session_state.vault:
        extracted = ""
        file_bytes = uploaded.read()
        with st.spinner("Reading medical document..."):
            if uploaded.type == "application/pdf":
                reader = PdfReader(BytesIO(file_bytes))
                extracted = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
                if not extracted:
                    try:
                        extracted = call_gemini_resilient(
                            client,
                            contents=[
                                types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                                (
                                    "Extract all medical information verbatim from this document "
                                    "(whether it is a lab report, X-ray/radiology report, pregnancy ultrasound, "
                                    "prescription, or discharge summary), including test names, measurements, "
                                    "impressions, and reference ranges."
                                ),
                            ],
                        ).strip()
                    except Exception as e:
                        st.error(f"Scanned PDF read error: {e}")
            elif uploaded.type.startswith("image/"):
                try:
                    img = Image.open(BytesIO(file_bytes)).convert("RGB")
                    img.thumbnail((1600, 1600))
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    extracted = call_gemini_resilient(
                        client,
                        contents=[
                            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                            (
                                "Examine this medical image or scanned document. If it is a printed report "
                                "(lab test, pregnancy ultrasound report, X-ray report, prescription), transcribe "
                                "all text, measurements, impressions, and reference ranges verbatim. If it is a "
                                "direct medical scan image (such as an X-ray or ultrasound), describe the visible "
                                "anatomical structures, notable visual observations, and any printed labels clearly "
                                "for educational explanation."
                            ),
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
        "Saved Documents",
        report_names,
        index=report_names.index(st.session_state.active_report),
    )
    if selected_report != st.session_state.active_report:
        st.session_state.active_report = selected_report
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption(f"🧠 Local ML Model: Random Forest ({ml_artifact['accuracy']}% val. accuracy)")

current_text = st.session_state.vault[st.session_state.active_report]
cache_key = f"{st.session_state.active_report}__{selected_lang}"

if cache_key not in st.session_state.analysis_cache:
    with st.spinner(f"Running ML triage & translating into {selected_lang}..."):
        try:
            st.session_state.analysis_cache[cache_key] = analyze_medical_document(
                client, current_text, selected_lang
            )
        except Exception:
            st.warning("⚠️ Free-tier daily AI limit reached on current key — switched to local offline parser mode.")
            st.session_state.analysis_cache[cache_key] = build_local_fallback_analysis(current_text)

data = st.session_state.analysis_cache[cache_key]
detected_type = data.get("document_type", "Medical Document")
ml_result = predict_ml_risk(current_text)

# --- HERO HEADER ---
st.markdown(
    f'<div class="hero-box">'
    f'<h1>🩺 MediPlain AI — Universal Medical Explainer</h1>'
    f'<p>Active File: <b>{st.session_state.active_report}</b> &nbsp;|&nbsp; Language: <b>{selected_lang}</b></p>'
    f'<div class="doc-pill">📄 Detected Type: {detected_type}</div>'
    f'</div>',
    unsafe_allow_html=True,
)
st.caption("⚠️ Educational health-literacy assistant only — not a medical diagnosis. Always consult a qualified doctor.")

tab1, tab2, tab3 = st.tabs([
    "🔬 Finding-by-Finding Breakdown",
    "💬 Voice & Text Chatbot",
    "📃 Extracted Document Text",
])

# --- TAB 1: NEAT FINDING-BY-FINDING EXPLAINER ---
with tab1:
    if ml_result:
        st.markdown(
            f'<div class="ml-card">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
            f'<span style="font-weight:700; color:#4c1d95; font-size:1rem;">🤖 Trained ML Biomarker Risk Screening (Random Forest Classifier)</span>'
            f'<span style="background:{ml_result["bg"]}; color:{ml_result["fg"]}; padding:4px 12px; border-radius:999px; font-weight:700; font-size:0.85rem;">'
            f'{ml_result["label"]} ({ml_result["confidence"]}% confidence)</span>'
            f'</div>'
            f'<p style="margin:6px 0 0 0; font-size:0.88rem; color:#475569;">'
            f'Evaluated {ml_result["matched_count"]} extracted biomarker(s) locally using <code>clinical_risk_model.pkl</code> (Validation Accuracy: {ml_result["accuracy"]}%).</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="summary-card">'
        f'<h4 style="margin:0 0 8px 0; color:#0284c7;">📋 Plain-Language Overview ({selected_lang})</h4>'
        f'<p style="margin:0; font-size:1.02rem; color:#1e293b; line-height:1.6;">{data.get("summary", "")}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if auto_voice:
        summary_audio = text_to_audio(data.get("summary", ""), lang_code)
        if summary_audio:
            st.audio(summary_audio, format="audio/mp3")

    st.subheader("🧪 Key Findings & Terms Explained Neatly")

    findings_list = data.get("findings", data.get("biomarkers", []))
    for idx, item in enumerate(findings_list):
        status = item.get("status", "Informational")
        s_low = status.lower()
        if s_low == "normal":
            badge_cls = "badge-normal"
        elif s_low == "borderline":
            badge_cls = "badge-borderline"
        elif s_low in ("high", "low"):
            badge_cls = "badge-high"
        else:
            badge_cls = "badge-info"

        ref_text = item.get("reference_range", "N/A")
        ref_html = f' <span style="color:#64748b; font-size:0.85rem;">(Ref: {ref_text})</span>' if ref_text and ref_text != "N/A" else ""

        card_html = (
            f'<div class="bio-card">'
            f'<div class="bio-header">'
            f'<span class="bio-title">{item.get("name", "Finding")}</span>'
            f'<div><span class="bio-val">{item.get("value", "")}</span>{ref_html} '
            f'<span class="{badge_cls}">{status}</span></div>'
            f'</div>'
            f'<p style="margin:6px 0;"><b>🔹 What It Is:</b> {item.get("what_it_is", "")}</p>'
            f'<p style="margin:6px 0;"><b>🔹 Why It Matters:</b> {item.get("why_it_matters", "")}</p>'
            f'<p style="margin:6px 0;"><b>🔹 What Your Result Means:</b> {item.get("what_result_means", "")}</p>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        if st.button(f"🔊 Listen to {item.get('name', 'Finding')}", key=f"btn_{cache_key}_{idx}"):
            spoken = f"{item.get('name', '')}. {item.get('what_it_is', '')} {item.get('what_result_means', '')}"
            term_audio = text_to_audio(spoken, lang_code)
            if term_audio:
                st.audio(term_audio, format="audio/mp3")
            else:
                st.warning("Audio readout unavailable for this text snippet.")

    questions = data.get("doctor_questions", [])
    if questions:
        q_html = "".join(f'<li style="margin-bottom:6px;">{q}</li>' for q in questions)
        st.markdown(
            f'<div class="doctor-box">'
            f'<h4 style="margin:0 0 10px 0; color:#0f766e;">🩺 Smart Questions to Ask Your Doctor</h4>'
            f'<ul style="margin:0; padding-left:20px; color:#1e293b;">{q_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

# --- TAB 2: VOICE & TEXT CHATBOT ---
with tab2:
    st.subheader(f"💬 Ask Questions About This {detected_type} ({selected_lang})")

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
            f"Listen to the patient's voice question and answer clearly in {selected_lang} based strictly on this medical document ({detected_type}):\n{current_text}"
        )
        display_user_msg = "🎙️ [Voice Question]"
    elif user_text:
        user_prompt_parts.append(
            f"Answer the patient's question clearly in {selected_lang} based strictly on this medical document ({detected_type}):\n{current_text}\n\nQuestion: {user_text}"
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
                except Exception:
                    fallback_reply = (
                        f"Your document ({detected_type}) contains these recorded details:\n\n{current_text}\n\n"
                        "*(Note: Live cloud AI chat quota is temporarily full—please update your GEMINI_API_KEY in Streamlit Secrets or try again shortly.)*"
                    )
                    st.write(fallback_reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": fallback_reply})

# --- TAB 3: RAW REPORT ---
with tab3:
    st.subheader("Extracted Document Text")
    st.text_area("Raw Content", current_text, height=320, label_visibility="collapsed")
