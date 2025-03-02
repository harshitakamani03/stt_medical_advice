import streamlit as st
from st_audiorec import st_audiorec
import openai
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------
# 1. Environment Setup
# ----------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------------------
# 2. Page Config & Custom CSS
# ----------------------------------------------------------------
st.set_page_config(page_title="Speech to Text", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    body {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
    }

    .wave {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 15rem;
        background: url('https://svgshare.com/i/pZ5.svg') no-repeat;
        background-size: cover;
        z-index: -1;
    }

    .main .block-container {
        max-width: 700px;
        margin-top: 6rem;
        margin-bottom: 4rem;
        padding: 2rem;
        background: rgba(255,255,255, 0.6); 
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.3);
    }

    h1 {
        text-align: center;
        font-size: 2rem !important;
        margin-bottom: 1rem !important;
        color: #333;
        font-weight: 600;
    }

    .stAudioRecorder {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    button[kind="primary"] {
        background: #0A8FE6 !important;
        border-radius: 0.4rem !important;
        border: none !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        padding: 0.7rem 1.5rem !important;
        margin-top: 1rem;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    button[kind="primary"]:hover {
        background: #076FB2 !important;
    }

    .stTextArea textarea {
        font-size: 0.9rem !important;
        border-radius: 0.5rem;
        border: 1px solid #ccc;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="wave"></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------
# 3. Whisper Transcription
# ----------------------------------------------------------------
def transcribe_whisper(audio_bytes: bytes) -> str:
    """Transcribe WAV bytes with OpenAI's Whisper API."""
    if not OPENAI_API_KEY:
        return "Missing OPENAI_API_KEY."

    openai.api_key = OPENAI_API_KEY
    try:
        # Save audio to a temporary file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        with open("temp_audio.wav", "rb") as audio_file:
            transcript_data = openai.Audio.transcribe("whisper-1", audio_file)

        return transcript_data["text"].strip()
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""


# ----------------------------------------------------------------
# 4. GPT Medical Advice from the Transcript
# ----------------------------------------------------------------
def get_medical_advice_from_transcript(transcript: str) -> str:
    """
    Given a free-form conversation or notes in 'transcript',
    ask GPT for bullet-point medical advice. 
    We ensure GPT suggests at least one drug with a dosage (or "dosage not specified").
    """
    # Make sure your OPENAI_API_KEY is set
    openai.api_key = OPENAI_API_KEY

    # Updated prompt with the new requirement:
    prompt_text = f"""
You have the following transcript of a conversation or medical notes:

{transcript}

Based on this, provide bullet-point medical advice with the exact headings below.
For each heading:
- Enclose the heading in double asterisks (e.g. **Heading**).
- List bullet points, each line beginning with '- '.
- If there's no content for a heading, write 'None.' under that heading.

**Most Likely Diagnosis**
- Provide the most likely diagnosis with a brief explanation.

**Other Possible Diagnoses**
- A ranked list of other possible diagnoses from most probable to least probable, with reasons.

**Suggested Tests**
- Provide recommended tests.

**Prognosis**
- Prognosis for the most likely and other possible conditions.

**Suggested Treatment Plan**
- Provide recommended drugs, with their typical dosages (e.g., mg per day).
- If you recommend a drug, you must include a dosage or state "dosage not specified."
- You MUST recommend at least one drug, even if hypothetical, and include a dosage (or "dosage not specified").
- Mention any contra-indications or relevant warnings.

**Case Summary**
(Short concluding summary)

If no data is available for a heading, write 'None.' in that section.
"""

    try:
        # GPT call
        response = openai.ChatCompletion.create(
            model="gpt-4", 
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant with no disclaimers needed."
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            temperature=0.7
        )
        advice = response.choices[0].message.content.strip()
        return advice
    except Exception as e:
        logging.error(f"Error in get_medical_advice_from_transcript: {e}")
        return "Error generating medical advice. Please try again."



# ----------------------------------------------------------------
# 5. Main Streamlit App
# ----------------------------------------------------------------
def main():
    st.title("Speech to Text")

    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Please configure it in your .env file.")
        return

    if "audio_bytes" not in st.session_state:
        st.session_state["audio_bytes"] = None

    if "full_transcript" not in st.session_state:
        st.session_state["full_transcript"] = ""

    st.caption("Click the microphone button to record audio.")
    audio_data = st_audiorec()

    if audio_data is not None:
        if audio_data != st.session_state["audio_bytes"]:
            st.session_state["audio_bytes"] = audio_data
            st.session_state["full_transcript"] = "" 

    if st.session_state["audio_bytes"] is not None and not st.session_state["full_transcript"]:
        with st.spinner("Transcribing..."):
            transcript = transcribe_whisper(st.session_state["audio_bytes"])
            st.session_state["full_transcript"] = transcript

    st.text_area(
        label="Transcript",
        value=st.session_state["full_transcript"],
        height=200
    )

    if st.button("Get Medical Advice"):
        if st.session_state["full_transcript"]:
            with st.spinner("Generating advice..."):
                advice = get_medical_advice_from_transcript(st.session_state["full_transcript"])
            st.subheader("Medical Advice")
            st.write(advice)
        else:
            st.warning("No transcript available.")


    if st.button("Clear"):
        st.session_state["audio_bytes"] = None
        st.session_state["full_transcript"] = ""
        st.experimental_rerun()

if __name__ == "__main__":
    main()
