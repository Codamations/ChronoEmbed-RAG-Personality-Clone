import os

def clean_discord_chat(raw_text: str, target_username: str) -> list[str]:
    cleaned_lines = []
    if not raw_text or not target_username:
        return cleaned_lines

    target_clean = target_username.lower().strip()
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    
    current_author_is_target = False

    for line in lines:
        line_str = line.strip()

        # Skip empty lines, borders, and headers
        if not line_str or line_str.startswith("=") or line_str.startswith("---") or line_str.startswith("Guild:") or line_str.startswith("Channel:"):
            continue

        # Header line detection: [Timestamp] username
        if line_str.startswith("[") and "]" in line_str:
            close_bracket_idx = line_str.find("]")
            after_bracket = line_str[close_bracket_idx + 1:].strip()
            
            parts = after_bracket.split(maxsplit=1)
            if parts:
                raw_author = parts[0].strip()
                author_clean = raw_author.lower().split("#")[0]

                # Check if this header belongs to target
                if target_clean in author_clean or author_clean in target_clean:
                    current_author_is_target = True
                    
                    # If message text exists on the SAME line as header
                    if len(parts) > 1:
                        content = parts[1].strip()
                        if content and not any(content.startswith(x) for x in ["Started a call", "Joined a call", "http", "<Attachment:", "{Attachments}"]):
                            cleaned_lines.append(content)
                else:
                    current_author_is_target = False

        # Multiline text (message sits on line(s) directly below header)
        elif current_author_is_target:
            if not any(line_str.startswith(x) for x in ["Started a call", "Joined a call", "http", "<Attachment:", "{Attachments}"]):
                cleaned_lines.append(line_str)

    print(f"[DEBUG] Final Extracted Lines for {target_username}: {len(cleaned_lines)}")
    return cleaned_lines


def process_and_export_chat(raw_text: str, target_username: str, output_filepath: str = "cleaned_target.txt") -> list[str]:
    cleaned_lines = clean_discord_chat(raw_text, target_username)

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))

    return cleaned_lines