import asyncio
from datetime import datetime, timezone
import logging

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# -----------------------------
# CONFIG
# -----------------------------

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_LLM_MODEL = "llama3.1:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mem-test")


# -----------------------------
# MAIN
# -----------------------------

async def main():
    log.info(">> Initializing Ollama LLM client")

    llm_config = LLMConfig(
        api_key="ollama",               # dummy value, required by interface
        model=OLLAMA_LLM_MODEL,
        small_model=OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    llm_client = OpenAIGenericClient(config=llm_config)

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="ollama",
            embedding_model=OLLAMA_EMBED_MODEL,
            embedding_dim=768,
            base_url=OLLAMA_BASE_URL,
        )
    )

    reranker = OpenAIRerankerClient(
        client=llm_client,
        config=llm_config,
    )

    log.info(">> Connecting to Neo4j via Graphiti")

    graph = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=reranker,
    )

    try:
        log.info(">> Building indices (safe to run multiple times)")
        await graph.build_indices_and_constraints()

        # -----------------------------
        # INGEST EPISODES
        # -----------------------------

        now = datetime.now(timezone.utc)

        episodes = [
            {
                "name": "creator-teo",
                "content": (
                    "The creator named Teo was born on February 15, 2005. "
                    "Teo is a developer and AI enthusiast who enjoys building local AI systems "
                    "and experimenting with advanced memory-based AI frameworks. "
                    "He is known for creating the chaotic gremlin model Ene, and spends time "
                    "integrating AI models with interactive applications."
                ),
            },
            {
                "name": "ene-birthday",
                "content": (
                    "The model named Ene was created on January 14. "
                    "Ene is a mischievous AI entity designed to interact with users in a playful "
                    "and chaotic manner, offering advice, teasing, and performing tasks. "
                    "Despite Ene’s bratty persona, it is capable of useful computation and memory-based reasoning. "
                    "Ene was implemented to demonstrate memory indexing."
                ),
            },
        ]

        for ep in episodes:
            log.info(f">> Adding episode: {ep['name']}")
            await graph.add_episode(
                name=ep["name"],
                episode_body=ep["content"],
                source=EpisodeType.text,
                source_description="manual memory test",
                reference_time=now,
            )

        # -----------------------------
        # QUERY MEMORY
        # -----------------------------

        log.info(">> Query: When is Teo's birthday?")
        results = await graph.search("When is Teo's birthday?")

        print("\n--- RESULTS ---")
        for r in results:
            print(f"FACT: {r.fact}")
            if r.valid_at:
                print(f"  valid_from: {r.valid_at}")
            if r.invalid_at:
                print(f"  valid_until: {r.invalid_at}")
            print("")

        log.info(">> Query: When was Ene created?")
        results = await graph.search("When was Ene created?")

        print("\n--- RESULTS ---")
        for r in results:
            print(f"FACT: {r.fact}")
            if r.valid_at:
                print(f"  valid_from: {r.valid_at}")
            if r.invalid_at:
                print(f"  valid_until: {r.invalid_at}")
            print("")

    finally:
        log.info(">> Closing graph connection")
        await graph.close()


if __name__ == "__main__":
    asyncio.run(main())
