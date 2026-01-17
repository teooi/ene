# mem.py
import asyncio
import logging
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# ======================================================
# CONFIG
# ======================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_LLM_MODEL = "llama3.1:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ======================================================
# LOGGING
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ene.mem")

# ======================================================
# MAIN
# ======================================================

async def main():
    log.info("🧱 Initializing Graphiti (memory builder)")

    llm_config = LLMConfig(
        api_key="ollama",
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

    graphiti = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=reranker,
    )

    try:
        # --------------------------------------------------
        # REQUIRED SETUP
        # --------------------------------------------------
        await graphiti.build_indices_and_constraints()

        # --------------------------------------------------
        # MEMORY EPISODES
        # --------------------------------------------------
        episodes = [
            {
                "name": "teo-identity",
                "content": "Teo Imoto-Tar is a person.",
            },
            {
                "name": "teo-birth",
                "content": "Teo Imoto-Tar was born on February 15, 2005.",
            },
            {
                "name": "teo-education",
                "content": (
                    "Teo Imoto-Tar is an undergraduate student studying "
                    "computer science and mathematics at UC San Diego."
                ),
            },
            {
                "name": "teo-work",
                "content": (
                    "Teo Imoto-Tar works as a research assistant at the "
                    "Neuroelectronics Lab in the Jacobs School of Engineering."
                ),
            },
            {
                "name": "teo-interests",
                "content": (
                    "Teo Imoto-Tar is interested in software engineering, "
                    "AI research, computer vision, and computational neuroscience."
                ),
            },
            {
                "name": "teo-hobbies",
                "content": (
                    "Outside of research, Teo Imoto-Tar makes music and likes capybaras."
                ),
            },
            {
                "name": "ene-creation",
                "content": "Ene is an AI assistant created by Teo Imoto-Tar on January 14, 2026.",
            },
        ]

        for i, ep in enumerate(episodes):
            log.info(f"🧠 Adding episode: {ep['name']}")
            await graphiti.add_episode(
                name=ep["name"],
                episode_body=ep["content"],
                source=EpisodeType.text,
                source_description="User-defined memory",
                reference_time=datetime.now(timezone.utc),
            )

        log.info("✅ Memory graph populated")

    finally:
        await graphiti.close()
        log.info("🔒 Graphiti connection closed")


if __name__ == "__main__":
    asyncio.run(main())
