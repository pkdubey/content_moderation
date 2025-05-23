import re
from typing import Tuple, Optional

def load_words(file_path: str) -> set:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using empty word list.")
        return set()

# Load word lists
banned_words = load_words('data/banned_words.txt')
political_words = load_words('data/political_words.txt')

def leet_to_text(text: str) -> str:
    """Convert leet speak and mixed characters to normal text"""
    leet_map = {
        '4': 'a', '@': 'a', '8': 'b', '3': 'e', '0': 'o',
        '5': 's', '7': 't', '1': 'i', '!': 'i', '$': 's', '+': 't'
    }
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text.lower())  # Remove special chars for detection
    return ''.join(leet_map.get(c, c) for c in cleaned)

def contains_banned_or_political(text: str) -> Tuple[bool, Optional[str]]:
    """Check for banned or political words with leet and obfuscation detection"""
    text_lower = text.lower()
    normalized_text = leet_to_text(text)  # Normalize whole text

    words_in_text = set(re.findall(r'[\w\'@$]+', text_lower))

    # 1. Exact match check
    if (found_words := banned_words.union(political_words) & words_in_text):
        return True, found_words.pop()

    # 2. Normalized word match
    normalized_words = {leet_to_text(word) for word in words_in_text}
    if (found_normalized := banned_words.union(political_words) & normalized_words):
        return True, found_normalized.pop()

    # 3. Substring scan in normalized text
    if (found_word := next((word for word in banned_words.union(political_words) if word in normalized_text), None)):
        return True, found_word

    return False, None