# mem.py (improved)
import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
# from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# --------------------------------------------
# Logging
# --------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mem")

# --------------------------------------------
# Helpers: flexible JSON loader + formatter
# --------------------------------------------
def load_raw_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonicalize_episodes(raw: Any) -> List[Dict]:
    """
    Return a list of episode dicts. Accepts:
      - list of objects each having key "episode" with content/metadata
      - top-level dict with key "episodes": [ ... ]
      - list of episode dicts (already canonical)
    """
    episodes: List[Dict] = []

    if isinstance(raw, dict) and "episodes" in raw and isinstance(raw["episodes"], list):
        raw_list = raw["episodes"]
    elif isinstance(raw, list):
        raw_list = raw
    else:
        # fallback: single episode wrapped in dict
        raw_list = [raw]

    for item in raw_list:
        # If wrapped as { "episode": {...} }
        if isinstance(item, dict) and "episode" in item and isinstance(item["episode"], dict):
            ep = item["episode"]
        elif isinstance(item, dict):
            ep = item
        else:
            # Unknown shape, skip
            log.warning("Skipping unknown item in JSON (not dict): %r", item)
            continue
        episodes.append(ep)
    return episodes


def build_episode_body_and_metadata(ep: Dict, idx: int) -> Tuple[str, Dict]:
    """
    Creates a human-readable episode_body for ingestion and a metadata dict.
    If ep contains a 'content' key, use it directly.
    Otherwise, synthesize a text from known fields (title, summary, time_range, traits).
    """
    # Use provided content if present
    if "content" in ep and isinstance(ep["content"], str) and ep["content"].strip():
        body = ep["content"].strip()
    else:
        # Compose a readable summary from structured fields
        parts = []
        if ep.get("title"):
            parts.append(f"Title: {ep.get('title')}")
        # Few structured fields often present in your example
        for field in ("type", "episode_id", "id"):
            if ep.get(field):
                parts.append(f"{field}: {ep.get(field)}")
        # time_range might be dict
        tr = ep.get("time_range")
        if isinstance(tr, dict):
            start = tr.get("start", "")
            end = tr.get("end", "")
            if start or end:
                parts.append(f"Time range: {start} — {end}")
        # summary / description
        if ep.get("summary"):
            parts.append(f"\nSummary:\n{ep.get('summary')}")
        # traits_inferred as list
        if ep.get("traits_inferred"):
            parts.append(f"\nTraits: {', '.join(map(str, ep.get('traits_inferred')))}")
        # fallback: include whole dict as json if nothing else
        if not parts:
            parts.append(json.dumps(ep, ensure_ascii=False, indent=2))
        body = "\n".join(parts)

    # metadata: preserve any metadata-like fields so they are searchable in the graph
    metadata = ep.get("metadata", {})
    # copy useful fields into metadata
    for k in ("episode_id", "id", "type", "title", "time_range", "confidence"):
        if ep.get(k) is not None:
            metadata[k] = ep[k]
    # store original object for traceability
    metadata["_raw"] = ep
    metadata["_import_index"] = idx
    return body, metadata


# --------------------------------------------
# Main ingestion
# --------------------------------------------
async def main():
    # CLI + env-based config
    parser = argparse.ArgumentParser(description="Ingest episodes into Graphiti.")
    parser.add_argument("--json", help="Path to JSON episodes", default=os.environ.get("GRAPHITI_EPISODES_PATH", "/Users/teoi/Documents/ene/toe_data/graphiti_ready.json"))
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "password"))
    parser.add_argument("--ollama-base", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--ollama-model", default=os.environ.get("OLLAMA_LLM_MODEL", "llama3.1:8b"))
    parser.add_argument("--embed-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    # parser.add_argument("--clear", action="store_true", help="Clear existing graph before ingest")
    args = parser.parse_args()

    json_path = Path(args.json)

    log.info("Initializing Graphiti")
    llm_config = LLMConfig(api_key=os.environ.get("OLLAMA_API_KEY", "ollama"), model=args.ollama_model, small_model=args.ollama_model, base_url=args.ollama_base)

    llm_client = OpenAIGenericClient(config=llm_config)

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            embedding_model=args.embed_model,
            # Do not hardcode embedding_dim unless you know it
            # embedding_dim=768,
            base_url=args.ollama_base,
        )
    )

    reranker = OpenAIRerankerClient(client=llm_client, config=llm_config)

    graphiti = Graphiti(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_pass,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=reranker,
    )

    try:
        await graphiti.build_indices_and_constraints()

        # if args.clear:
        #     log.warning("Clearing existing memory graph")
        #     await clear_data(graphiti.driver)

        raw = load_raw_json(json_path)
        episodes_raw = canonicalize_episodes(raw)
        log.info("Loaded %d raw episodes", len(episodes_raw))

        added = 0
        for idx, ep in enumerate(episodes_raw):
            body, metadata = build_episode_body_and_metadata(ep, idx)

            # warn about very long content (optional)
            if len(body) > 50_000:
                log.warning("Episode %s is very long (%d chars) — consider truncating/splitting", metadata.get("episode_id", idx), len(body))

            try:
                await graphiti.add_episode(
                    name=f"json-episode-{idx}",
                    episode_body=body,
                    source=EpisodeType.text,
                    source_description="Imported from JSON",
                    reference_time=datetime.now(timezone.utc),
                )
                added += 1
                log.info("Ingested episode %d (%s)", idx, metadata.get("title", metadata.get("episode_id", str(idx))))
            except Exception as e:
                log.exception("Failed to ingest episode idx=%d id=%r: %s", idx, metadata.get("episode_id"), e)

        log.info("Ingestion complete. %d/%d episodes added.", added, len(episodes_raw))

    finally:
        # make sure to close
        try:
            await graphiti.close()
        except Exception:
            log.exception("Error closing Graphiti connection")

if __name__ == "__main__":
    asyncio.run(main())
