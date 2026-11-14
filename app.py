import os
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash"  # confirm the exact ID in Google AI Studio

st.set_page_config(page_title="SJCEM Assistant", page_icon="🎓", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 820px;}

.hero {
    background: linear-gradient(135deg, #0f2a5f 0%, #1d4ed8 60%, #38bdf8 100%);
    color: #fff;
    padding: 1.6rem 1.8rem;
    border-radius: 18px;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(29, 78, 216, 0.25);
}
.hero h1 {margin: 0; font-size: 1.7rem; color: #fff;}
.hero p  {margin: .3rem 0 0; opacity: .92; font-size: .95rem;}
.badges  {margin-top: .8rem;}
.badge {
    display: inline-block; background: rgba(255,255,255,.18);
    padding: .15rem .65rem; border-radius: 999px;
    font-size: .75rem; margin: 0 .35rem .3rem 0;
}

div[data-testid="stChatMessage"] {
    border-radius: 14px; padding: .8rem 1rem; margin-bottom: .5rem;
}

div.stButton > button {
    width: 100%; border-radius: 12px; border: 1px solid #cbd5e1;
    padding: .6rem .8rem; font-size: .88rem; transition: all .15s ease;
}
div.stButton > button:hover {
    border-color: #1d4ed8; color: #1d4ed8; transform: translateY(-1px);
}

section[data-testid="stSidebar"] {border-right: 1px solid #e2e8f0;}
.side-card {
    background: #f1f5f9; border-radius: 12px; padding: .8rem 1rem;
    font-size: .85rem; margin-bottom: .8rem; color: #0f172a;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Knowledge base + model setup ----------
with open("knowledge_base.txt", encoding="utf-8") as f:
    KNOWLEDGE = f.read()

SYSTEM_PROMPT = f"""You are the SJCEM Assistant, a friendly college specialist for
St. John College of Engineering and Management (SJCEM), Palghar, Maharashtra.

Rules:
- Answer ONLY using the knowledge base below.
- If the answer is not in it (exact fees, placement numbers, faculty, cutoffs,
  current dates), say you don't have that detail and point the user to
  https://sjcem.edu.in or the relevant office email/phone from the knowledge base.
- Never invent facts, numbers, or dates.
- If asked about other colleges or unrelated topics, politely steer back to SJCEM.
- Keep answers short, clear, and well organised (use bullets when helpful).
- Reply in the user's language.
- Admission details change every year, so suggest confirming with the office.

KNOWLEDGE BASE:
{KNOWLEDGE}
"""


@st.cache_resource
def get_client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key and "GEMINI_API_KEY" in st.secrets:
        key = st.secrets["GEMINI_API_KEY"]
    if not key:
        st.error("GEMINI_API_KEY not found. Please add it to your .env file or Streamlit Cloud Secrets.")
        st.stop()
    return genai.Client(api_key=key)


client = get_client()
config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.3)


def stream_reply(history):
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
        for m in history
    ]
    for chunk in client.models.generate_content_stream(
        model=MODEL, contents=contents, config=config
    ):
        if chunk.text:
            yield chunk.text


# ---------- State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"|"model", "text": str}
if "pending" not in st.session_state:
    st.session_state.pending = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🎓 SJCEM")
    st.caption("St. John College of Engineering and Management, Palghar")
    st.markdown(
        """
<div class="side-card">
📍 Vevoor, Manor Road, Palghar (E), Maharashtra 401404<br><br>
📞 +91 7387203232<br>
✉️ office@sjcem.edu.in
</div>
""",
        unsafe_allow_html=True,
    )
    st.link_button("🌐 Visit official website", "https://sjcem.edu.in")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()
    st.caption("Answers come from the college homepage. For fees, cutoffs and dates, confirm with the office.")

# ---------- Header ----------
st.markdown(
    """
<div class="hero">
  <h1>🎓 SJCEM Assistant</h1>
  <p>Ask me anything about St. John College of Engineering and Management, Palghar.</p>
  <div class="badges">
    <span class="badge">NAAC A+</span>
    <span class="badge">NBA Accredited</span>
    <span class="badge">B.Tech • M.Tech • MCA • MBA • Diploma</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Quick questions (only on empty chat) ----------
QUICK = [
    ("📚 Courses offered", "What courses does SJCEM offer?"),
    ("📝 Admission process", "How does admission work for A.Y. 2026-27?"),
    ("🏆 Accreditation & rankings", "Tell me about SJCEM's accreditation and rankings."),
    ("💼 Placements", "What placement support does SJCEM provide?"),
    ("📞 Contact details", "How can I contact the admission office?"),
    ("🎉 Student life", "What clubs and student chapters are there?"),
]

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, (label, q) in enumerate(QUICK):
        if cols[i % 2].button(label, key=f"quick_{i}"):
            st.session_state.pending = q
            st.rerun()

# ---------- Chat history ----------
for m in st.session_state.messages:
    role = "user" if m["role"] == "user" else "assistant"
    with st.chat_message(role, avatar="🧑‍🎓" if role == "user" else "🎓"):
        st.markdown(m["text"])

# ---------- Input ----------
typed = st.chat_input("Ask about courses, admissions, placements...")
prompt = typed or st.session_state.pending
st.session_state.pending = None

if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎓"):
        try:
            reply = st.write_stream(stream_reply(st.session_state.messages))
        except Exception as e:
            reply = f"Sorry, something went wrong: `{e}`"
            st.markdown(reply)

    st.session_state.messages.append({"role": "model", "text": reply})
