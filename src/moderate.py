from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.filter import contains_banned_or_political
from typing import Optional

# Load fine-tuned model
model_path = "models/transformer_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}") from e

def classify_text(text: str, threshold: float = 0.7) -> int:
    """Classify text with confidence threshold"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    toxic_prob = probs[0][1].item()  # Probability of being toxic
    
    return 1 if toxic_prob > threshold else 0

def moderate_content(text: str) -> str:
    """Moderate content with multi-layer filtering"""
    if not text.strip():
        return "❌ Blocked: Empty message"
    
    # First check for banned/political words
    flagged, word = contains_banned_or_political(text)
    if flagged:
        return f"❌ Blocked: contains banned/political term - '{word}'"

    # Then check for toxic language (with confidence threshold)
    try:
        if classify_text(text) == 1:
            return "❌ Blocked: contains toxic/impolite language"
    except Exception as e:
        print(f"Error in AI classification: {e}")
        return "⚠️ Warning: Could not complete content analysis"

    return "✅ Approved: Clean content"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(moderate_content(' '.join(sys.argv[1:])))
    else:
        print("Please provide text to moderate as argument")