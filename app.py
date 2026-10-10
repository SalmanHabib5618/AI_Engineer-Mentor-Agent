import streamlit as st

from ds_mentor_agent import answer
from memory_store import load_messages, save_message, clear_all, get_progress, set_progress

st.set_page_config(page_title="AI Engineer Mentor", page_icon="🧭", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #FBFAF7; }
[data-testid="stChatMessage"] { border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧭 Your Mentor")
    st.caption("Your partner from beginner to AI Engineer")
    mode = st.selectbox(
        "Model mode",
        ["Automatic", "API (Gemini)", "Local (Ollama)"],
        help="Automatic prefers a local Ollama server when available, then uses Gemini only if a key is configured."
    )
    st.caption("Automatic mode does not make paid API calls unless you configure a provider key and choose to use it.")
    st.divider()
    st.subheader("Learning progress")
    current_stage = st.text_input(
        "Current stage",
        value=get_progress("current_stage", "Stage 1 — Python foundations"),
    )
    if st.button("Save stage", use_container_width=True):
        set_progress("current_stage", current_stage.strip())
        st.success("Stage saved on this computer.")
    st.caption("Progress is stored locally in a SQLite database.")
    st.divider()
    if st.button("Clear saved chat and progress", use_container_width=True):
        clear_all()
        st.session_state.messages = []
        st.success("Local chat and progress cleared.")
        st.rerun()

st.title("AI Engineer Mentor")
st.write("Your personal learning partner, teacher, and guide — from scratch to job-ready.")
st.info("Guided learning • Local progress memory • API/local provider interface")

if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask, learn, debug, plan a project, or just say hello…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    recent = st.session_state.messages[:-1]
    with st.chat_message("assistant"):
        with st.spinner("Thinking with you…"):
            response, route, provider_name = answer(prompt, mode=mode, recent_history=recent)
        st.markdown(response)
        st.caption(f"Mode: {route.conversation_mode} · Task: {route.intent} · Provider: {provider_name}")

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message("assistant", response)
