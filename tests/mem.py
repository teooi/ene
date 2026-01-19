import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
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
# Helpers: Ene-Specific Parsers
# --------------------------------------------
def load_raw_json(path: Path) -> List[Dict]:
    if not path.exists():
        # Help debugging if path is wrong
        log.error(f"Looking for file at: {path.absolute()}")
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    if isinstance(data, list):
        return data
    raise ValueError("JSON must be a list of episodes or a dict with an 'episodes' key.")

def format_ene_episode(ep: Dict) -> Tuple[str, datetime]:
    timestamp_str = ep.get("timestamp")
    reference_time = datetime.now(timezone.utc)
    
    if timestamp_str:
        try:
            reference_time = datetime.fromisoformat(timestamp_str)
            if reference_time.tzinfo is None:
                reference_time = reference_time.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    lines = []
    lines.append(f"EVENT SUMMARY: {ep.get('summary', 'Unknown Event')}")
    lines.append(f"CATEGORY: {ep.get('type', 'General')}")
    lines.append(f"ROAST LEVEL: {ep.get('roast_potential', 'None')}")
    lines.append("\n--- EVENT DETAILS (AMMUNITION) ---")
    lines.append(ep.get('ammunition', ''))
    
    if ep.get('ene_internal_note'):
        lines.append("\n--- INTERNAL CONTEXT ---")
        lines.append(ep.get('ene_internal_note'))
    
    keywords = ep.get('trigger_keywords', [])
    if keywords:
        lines.append(f"\nKEYWORDS: {', '.join(keywords)}")

    return "\n".join(lines), reference_time

# --------------------------------------------
# Main Ingestion
# --------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Ingest Ene episodes into Graphiti.")
    
    # --- UPDATED DEFAULTS HERE ---
    parser.add_argument(
        "--json", 
        help="Path to JSON episodes", 
        default="/Users/teoi/Documents/ene/toe_data/graphiti_ready.json"
    )
    # Changed localhost -> 127.0.0.1 to fix your connection error permanently
    parser.add_argument(
        "--neo4j-uri", 
        default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    )
    
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "password"))
    parser.add_argument("--ollama-base", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--ollama-model", default=os.environ.get("OLLAMA_LLM_MODEL", "llama3.1:8b"))
    parser.add_argument("--embed-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    args = parser.parse_args()

    json_path = Path(args.json)

    log.info("Initializing Graphiti Client...")
    
    llm_config = LLMConfig(
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"), 
        model=args.ollama_model, 
        base_url=args.ollama_base
    )
    llm_client = OpenAIGenericClient(config=llm_config)

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            embedding_model=args.embed_model,
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
        log.info("Building indices...")
        await graphiti.build_indices_and_constraints()

        log.info(f"Loading episodes from {json_path}...")
        episodes_raw = load_raw_json(json_path)
        
        log.info(f"Found {len(episodes_raw)} episodes. Starting ingestion...")

        added = 0
        for idx, ep in enumerate(episodes_raw):
            episode_id = ep.get("id", f"ep_{idx}")
            body, ref_time = format_ene_episode(ep)

            try:
                await graphiti.add_episode(
                    name=episode_id,
                    episode_body=body,
                    source=EpisodeType.text,
                    source_description="Ene Interaction History",
                    reference_time=ref_time,
                )
                added += 1
                log.info(f"[{idx+1}/{len(episodes_raw)}] Ingested '{episode_id}'")
                
            except Exception as e:
                log.exception(f"Failed to ingest episode {episode_id}: {e}")

        log.info(f"Ingestion complete. {added}/{len(episodes_raw)} episodes added.")

    except Exception as e:
        log.critical(f"Critical error during execution: {e}")
    finally:
        try:
            await graphiti.close()
            log.info("Connection closed.")
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())