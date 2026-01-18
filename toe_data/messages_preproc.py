import json
import re

INPUT = "messages.json"
OUTPUT = "my_messages_clean.json"

REACTION_PATTERNS = [
    r"^(liked|loved|emphasized|disliked|laughed at) an (image|photo|video|attachment)$",
    r"^(liked|loved|emphasized|disliked|laughed at) a (message|photo|video)$",
    r"^(liked|loved|emphasized|disliked|laughed at)\s+“.*”$",
    r"^(liked|loved|emphasized|disliked|laughed at)\s+\".*\"$",
]

def is_reaction(text: str) -> bool:
    t = text.strip().lower()
    return any(re.match(p, t) for p in REACTION_PATTERNS)

def is_good(text: str) -> bool:
    t = text.strip()
    return (
        len(t) >= 5 and
        not t.isdigit() and
        not is_reaction(t)
    )

with open(INPUT) as f:
    messages = json.load(f)

clean = [
    {
        "timestamp": m["timestamp"],
        "text": m["text"].strip()
    }
    for m in messages
    if (
        m.get("is_from_me") == 1 and
        m.get("text") and
        is_good(m["text"])
    )
]

with open(OUTPUT, "w") as f:
    json.dump(clean, f, indent=2, ensure_ascii=False)

print(f"Saved {len(clean)} cleaned messages → {OUTPUT}")

