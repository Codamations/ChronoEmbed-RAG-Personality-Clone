import re

def analyze_personality_style(cleaned_lines: list[str]) -> dict:
    """
    Analyzes list of strings for typing habits and returns percentage metrics.
    """
    if not cleaned_lines:
        return {
            "lowercase_ratio": 0.5,
            "punctuation_ratio": 0.5,
            "avg_length": 10,
            "emoji_frequency": 0.1
        }

    total_lines = len(cleaned_lines)
    lowercase_count = sum(1 for line in cleaned_lines if line.islower())
    punctuation_count = sum(1 for line in cleaned_lines if re.search(r"[.,!?]$", line))
    total_words = sum(len(line.split()) for line in cleaned_lines)
    emoji_count = sum(1 for line in cleaned_lines if re.search(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]", line))

    return {
        "lowercase_ratio": round(lowercase_count / total_lines, 2),
        "punctuation_ratio": round(punctuation_count / total_lines, 2),
        "avg_length": round(total_words / total_lines, 1),
        "emoji_frequency": round(emoji_count / total_lines, 2)
    }

def generate_style_rules(stats: dict) -> str:
    """
    Translates numeric style stats into explicit system prompt instructions.
    """
    rules = []
    
    if stats.get("lowercase_ratio", 0) > 0.6:
        rules.append("- Type predominantly in ALL LOWERCASE. Avoid standard capitalization unless emphasizing.")
    elif stats.get("lowercase_ratio", 0) < 0.2:
        rules.append("- Use standard capitalization rules consistently.")
        
    if stats.get("punctuation_ratio", 0) < 0.3:
        rules.append("- Rarely use ending punctuation (periods, commas at end of messages). Keep sentences raw.")
    
    avg_len = stats.get("avg_length", 10)
    if avg_len < 6:
        rules.append("- Keep responses short, punchy, and terse (1-8 words on average).")
    elif avg_len > 15:
        rules.append("- Feel free to write longer, detailed thoughts across multiple clauses.")

    if stats.get("emoji_frequency", 0) < 0.05:
        rules.append("- DO NOT use emojis unless extremely specific context calls for it.")

    return "\n".join(rules) if rules else "- Match standard casual typing style."