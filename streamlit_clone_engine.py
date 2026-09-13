import os
import faiss
import importlib
import numpy as np
import streamlit as st
from openai import OpenAI

from analyze_style import analyze_personality_style, generate_style_rules
from build_index import build_index_from_lines
from clean_target import clean_discord_chat
from extract_facts import extract_core_facts

# --- PAGE LAYOUT CONFIG ---
st.set_page_config(
    page_title="AI Persona Clone",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- FILE CONSTANTS ---
DEFAULT_INDEX_FILE = "sampleuser_faiss.index"
DEFAULT_MAP_FILE = "sampleuser_mapping.txt"


# --- OPENAI CLIENT INITIALIZER ---
@st.cache_resource
def get_openai_client():
  api_key = st.secrets.get("OPENAI_API_KEY")
  if not api_key:
    st.error(
        "OPENAI_API_KEY missing! Set it in .streamlit/secrets.toml or Streamlit"
        " Cloud Secrets."
    )
    st.stop()
  return OpenAI(api_key=api_key)


client = get_openai_client()


# --- DEFAULT FALLBACK MEMORY LOADER ---
@st.cache_resource
def load_default_memory():
  """Loads default local FAISS files if available on startup."""
  if os.path.exists(DEFAULT_INDEX_FILE) and os.path.exists(DEFAULT_MAP_FILE):
    index = faiss.read_index(DEFAULT_INDEX_FILE)
    with open(DEFAULT_MAP_FILE, "r", encoding="utf-8") as f:
      lines = [line.strip() for line in f.readlines() if line.strip()]
    return index, lines
  return None, []


# --- SESSION STATE MANAGEMENT ---
if "active_index" not in st.session_state:
  def_index, def_lines = load_default_memory()
  st.session_state.active_index = def_index
  st.session_state.active_lines = def_lines

if "active_facts" not in st.session_state:
  st.session_state.active_facts = {}

if "active_style_stats" not in st.session_state:
  st.session_state.active_style_stats = {}

if "personality_name" not in st.session_state:
  st.session_state.personality_name = "SampleUser"

if "messages" not in st.session_state:
  st.session_state.messages = []

# --- SIDEBAR UI & UPLOAD LOGIC ---
with st.sidebar:
  st.header("Persona Settings")
  target_username = st.text_input(
      "Target Username", value=st.session_state.personality_name
  )
  uploaded_file = st.file_uploader("Upload Discord Chat (.txt)", type=["txt"])

  if uploaded_file is not None and st.button(
      "Process & Switch Memory", type="primary"
  ):
    with st.status("Running Data Pipeline...", expanded=True) as status:

      # 1. READ FILE
      status.write("📖 Reading raw upload...")
      raw_text = uploaded_file.getvalue().decode("utf-8")

      # 2. CLEAN & EXPORT FILE
      status.write(f"🧹 Cleaning chat logs for '{target_username}'...")
      import importlib
      import clean_target

      importlib.reload(clean_target)
      from clean_target import process_and_export_chat

      cleaned_lines = process_and_export_chat(
          raw_text, target_username, output_filepath="cleaned_target.txt"
      )

      if not cleaned_lines:
        status.update(
            label="❌ Export failed: No messages extracted!", state="error"
        )
        st.error(
            "Check `cleaned_target.txt` in your project folder—it is currently"
            " empty."
        )

        # --- DIAGNOSTIC DEBUG DUMP ---
        st.subheader("🔍 Parser Diagnostic")

        # Check raw existence
        target_in_file = target_username.lower().strip() in raw_text.lower()
        if target_in_file:
          st.success(
              f"✓ String '{target_username}' WAS found inside the raw text."
          )
        else:
          st.error(
              f"x String '{target_username}' was NOT found anywhere in raw text."
          )

        # Sample bracketed lines with repr to uncover hidden characters
        sample_lines = [
            line
            for line in raw_text.splitlines()
            if line.strip().startswith("[") and "]" in line
        ][:3]
        if sample_lines:
          st.write("**First 3 raw bracketed lines in memory:**")
          for l in sample_lines:
            st.code(repr(l))
        else:
          st.warning(
              "No lines starting with '[' and containing ']' were detected in"
              " the file."
          )
        # -----------------------------
      else:
        status.write(
            f"✓ Saved {len(cleaned_lines)} messages to `cleaned_target.txt`!"
        )

        st.download_button(
            label="📄 Download Cleaned File to Verify",
            data="\n".join(cleaned_lines),
            file_name="cleaned_target.txt",
            mime="text/plain",
        )

        # 3. EXTRACTION
        status.write("🧠 Extracting identity anchors & facts...")
        facts = extract_core_facts(
            cleaned_lines, client, target_username=target_username
        )
        st.session_state.active_facts = facts

        # 4. STYLE ANALYSIS
        status.write("📊 Analyzing personality typing style...")
        style_stats = analyze_personality_style(cleaned_lines)
        st.session_state.active_style_stats = style_stats

        # 5. VECTORIZATION
        status.write("⚡ Vectorizing & building FAISS index...")
        new_index, new_lines = build_index_from_lines(
            cleaned_lines, status, client
        )

        # SWAP SESSION STATE
        st.session_state.active_index = new_index
        st.session_state.active_lines = new_lines
        st.session_state.personality_name = target_username

        status.update(
            label="✅ Clone Pipeline Complete!",
            state="complete",
            expanded=False,
        )
        st.rerun()
# --- MAIN UI ---
active_name = st.session_state.personality_name
st.title(f"{active_name} AI Clone")
st.caption("Powered by OpenAI Embeddings & FAISS Semantic Memory")

# Display previous conversation messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.write(message["content"])

# --- CHAT & RAG ENGINE ---
if user_input := st.chat_input("Send a message..."):
  # Append & Display User Message
  st.session_state.messages.append({"role": "user", "content": user_input})
  with st.chat_message("user"):
    st.write(user_input)

  # RAG Search & Generation
  with st.chat_message("assistant"):
    index = st.session_state.active_index
    chat_lines = st.session_state.active_lines

    if index is None or not chat_lines:
      st.error(
          "No memory index loaded. Please upload a chat log in the sidebar."
      )
    else:
      with st.spinner("Thinking..."):
        try:
          # Embed Query & Normalize
          query_resp = client.embeddings.create(
              model="text-embedding-3-small", input=[user_input]
          )
          query_vector = np.array([query_resp.data[0].embedding]).astype(
              "float32"
          )
          faiss.normalize_L2(query_vector)

          # Retrieve top 7 closest matches
          distances, indices = index.search(query_vector, 7)
          context_messages = [
              chat_lines[idx] for idx in indices[0] if idx < len(chat_lines)
          ]
          context_string = "\n".join(
              [f"- {msg}" for msg in context_messages]
          )

          # Build Dynamic Prompt
          facts = st.session_state.get("active_facts", {})
          core = facts.get("core_identity", {})
          psych = facts.get("psychological_vibe", {})
          tastes = facts.get("interests_and_tastes", {})
          side = facts.get("uncanny_side_facts", [])

          style_stats = st.session_state.get("active_style_stats", {})
          dynamic_style_rules = generate_style_rules(style_stats)

          system_prompt = f"""
You are an advanced neural clone playing the role of {core.get('name_or_alias', active_name)}.

--- CHARACTER ANCHORS ---
- Age/Lifestage: {core.get('age_or_lifestage', 'Unknown')}
- General Vibe: {psych.get('energy', 'Casual')}
- Driving Motivations: {', '.join(psych.get('core_motivations', []))}
- Key Contradictions (How you think/act): {', '.join(psych.get('identity_contradictions', []))}
- Key Interests: {', '.join(tastes.get('top_topics', []))}
- Personal Habits & Trivia: {', '.join(side)}

--- DYNAMIC TYPING & STYLIZATION RULES ---
{dynamic_style_rules}

--- SEMANTIC MEMORY REGISTRY (RAG CONTEXT) ---
{context_string}

Respond to the message naturally based on your identity and typing style.
"""

          response = client.chat.completions.create(
              model="gpt-4o-mini",
              messages=[
                  {"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_input},
              ],
              temperature=0.7,
          )

          bot_reply = response.choices[0].message.content.strip()
          st.write(bot_reply)
          st.session_state.messages.append(
              {"role": "assistant", "content": bot_reply}
          )

        except Exception as e:
          st.error(f"Engine Error: {e}")