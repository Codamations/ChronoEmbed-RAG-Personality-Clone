import os
import faiss

# Native TOML parsing in Python 3.11+
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from openai import OpenAI
from clean_target import clean_discord_chat
from build_index import build_index_from_lines

def main():
    # Load key directly from secrets file
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        raise FileNotFoundError("Could not find .streamlit/secrets.toml file.")

    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        api_key = secrets.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .streamlit/secrets.toml")

    client = OpenAI(api_key=api_key)

    # Load raw chat file
    if not os.path.exists("sample_chat.txt"):
        raise FileNotFoundError("Please ensure 'sample_chat.txt' exists in project root.")

    with open("sample_chat.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Cleaning chat data...")
    cleaned_lines = clean_discord_chat(raw_text, target_username="SampleUser")
    print(f"Extracted {len(cleaned_lines)} lines for SampleUser.")

    print("Generating FAISS index and embeddings...")
    index, lines = build_index_from_lines(cleaned_lines, None, client)

    # Save binaries
    faiss.write_index(index, "sampleuser_faiss.index")
    with open("sampleuser_mapping.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("✅ SUCCESS: Generated sampleuser_faiss.index and sampleuser_mapping.txt")

if __name__ == "__main__":
    main()