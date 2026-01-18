"""
Preprocess cleaned iMessage data into Graphiti-ready memory episodes.

Input:
  - my_messages_clean.json

Output:
  - episode_candidates.json
  - graphiti_ready.json

Pipeline:
  1. Window messages into chunks
  2. Build episode candidates
  3. Use raw text as summary (no LLM)
  4. Prune low-confidence data (empty since no LLM)
  5. Emit Graphiti-ready objects
"""

import os
import json
from typing import List, Dict

# -----------------------
# Configuration
# -----------------------

INPUT_FILE = "my_messages_clean.json"
EPISODE_CANDIDATES_FILE = "episode_candidates.json"
GRAPHITI_READY_FILE = "graphiti_ready.json"

WINDOW_SIZE = 4

# -----------------------
# Step 1: Load messages
# -----------------------

with open(INPUT_FILE, "r") as f:
    messages = json.load(f)

assert messages, "No messages found"

# -----------------------
# Step 2: Window messages
# -----------------------

chunks = [messages[i:i + WINDOW_SIZE] for i in range(0, len(messages), WINDOW_SIZE)]

# -----------------------
# Step 3: Build episode candidates
# -----------------------

def make_episode_candidate(chunk: List[Dict], idx: int) -> Dict:
    return {
        "episode_id": f"imessage_{idx}",
        "source": "imessage",
        "episode_type": "persona_bootstrap",
        "time_range": {
            "start": chunk[0]["timestamp"],
            "end": chunk[-1]["timestamp"],
        },
        "raw_text": "\n".join(m["text"] for m in chunk),
    }

episode_candidates = [make_episode_candidate(chunk, i) for i, chunk in enumerate(chunks)]

with open(EPISODE_CANDIDATES_FILE, "w") as f:
    json.dump(episode_candidates, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(episode_candidates)} episode candidates → {EPISODE_CANDIDATES_FILE}")

# -----------------------
# Step 4: Prune / normalize (no LLM, so just pass)
# -----------------------

def prune_extraction(extraction: Dict) -> Dict:
    # no traits, preferences, facts yet
    return {
        "summary": extraction.get("summary", "").strip(),
        "persona_traits": [],
        "preferences": [],
        "facts": [],
    }

# -----------------------
# Step 5: Build Graphiti-ready objects
# -----------------------

graphiti_ready = []

for ep in episode_candidates:
    extraction = {
        "summary": ep["raw_text"][:200]  # take first 200 chars as fallback
    }
    extraction = prune_extraction(extraction)

    graphiti_ready.append({
        "episode": {
            "episode_type": "MEMORY",
            "content": extraction["summary"],
            "metadata": {
                "source": ep["source"],
                "time_range": f"{ep['time_range']['start']} → {ep['time_range']['end']}",
            },
        },
        "persona_traits": extraction["persona_traits"],
        "preferences": extraction["preferences"],
        "facts": extraction["facts"],
    })

with open(GRAPHITI_READY_FILE, "w") as f:
    json.dump(graphiti_ready, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(graphiti_ready)} Graphiti-ready episodes → {GRAPHITI_READY_FILE}")
