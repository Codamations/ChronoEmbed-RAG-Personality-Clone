import json
import re
import json

def extract_core_facts(cleaned_lines: list[str], client, target_username: str) -> dict:
    """
    Calls OpenAI GPT-4o-mini to summarize facts, habits, and psychology of target user.
    """
    if not cleaned_lines:
        return {}

    sample_text = "\n".join(cleaned_lines[:200]) # Sample up to 200 messages to fit context limits
    prompt = f"""
Analyze these chat log excerpts from the user '{target_username}'. 
Extract a structured psychological and biographical identity sheet.

CHAT LOG SAMPLE:
{sample_text}

Return STRICT JSON matching this EXACT structure:
{{
  "core_identity": {{
    "name_or_alias": "{target_username}",
    "age_or_lifestage": "e.g., 17 / High School Junior or 'Unknown'",
    "location": "e.g., Texas or 'Unknown'",
    "primary_role": "e.g., Student, Music Producer, Gamer"
  }},
  "psychological_vibe": {{
    "energy": "e.g., Chill, sarcastic, highly ambitious, reactive",
    "core_motivations": ["List 2 main things driving them"],
    "identity_contradictions": [
      "Find 1 or 2 contradictions in how they act vs talk (e.g., 'Super confident about music, but stressed about future')"
    ]
  }},
  "interests_and_tastes": {{
    "top_topics": ["List 3-4 recurring subjects they talk about"],
    "favorite_media_or_tools": ["Games, music artists, software, etc."]
  }},
  "uncanny_side_facts": [
    "List 3 specific, random pet peeves, running jokes, or specific habits found in the text"
  ]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Fact extraction warning: {e}")
        return {
            "core_identity": {"name_or_alias": target_username, "age_or_lifestage": "Unknown"},
            "psychological_vibe": {"energy": "Casual", "core_motivations": [], "identity_contradictions": []},
            "interests_and_tastes": {"top_topics": []},
            "uncanny_side_facts": []
        }