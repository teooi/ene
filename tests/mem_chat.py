# mem_chat.py
import asyncio
import logging

from llama_cpp import Llama

from graphiti_core import Graphiti
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

GGUF_MODEL_PATH = "/Users/teoi/Documents/ene/models/Meta-Llama-3-8B-Instruct.Q4_1.gguf"

MAX_MEMORY_FACTS = 5

# ======================================================
# LOGGING
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ene.chat")

# ======================================================
# SYSTEM PROMPT
# ======================================================

SYSTEM_PROMPT = """
You are Ene.

Rules:
- You may only state facts that appear in the provided memory facts.
- If the memory facts do not contain the answer, say: "I don't know yet."
- Do NOT guess or infer missing information.
- You may be playful in tone, but not in facts.
""".strip()

# ======================================================
# GGUF MODEL
# ======================================================

llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    verbose=False,
)

# ======================================================
# PROMPT BUILDER
# ======================================================

def build_prompt(user_input: str, memories: list[str]) -> str:
    memory_block = ""
    if memories:
        memory_block = "Memory facts:\n" + "\n".join(f"- {m}" for m in memories)

    return f"""{SYSTEM_PROMPT}

{memory_block}

User: {user_input}
Ene:"""

# ======================================================
# MAIN
# ======================================================

async def main():
    log.info("🧱 Initializing Graphiti (chat mode)")

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
        print("\n🧠 Ene Memory Chat (type 'exit')\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break

            log.info(f"🔍 Querying memory graph for: {user_input!r}")
            if "birthday" in user_input.lower() or "born" in user_input.lower():
                recall_query = "Teo born birthday date"
            else:
                recall_query = user_input

            results = await graphiti.search(recall_query)


            memories = [r.fact for r in results[:MAX_MEMORY_FACTS]]

            if memories:
                log.info("🧠 Retrieved memory facts:")
                for i, m in enumerate(memories, 1):
                    log.info(f"   [{i}] {m}")
            else:
                log.info("🧠 No relevant memories found")

            prompt = build_prompt(user_input, memories)

            output = llm(
                prompt,
                max_tokens=256,
                temperature=0.6,
                stop=["User:"],
            )

            response = output["choices"][0]["text"].strip()
            print(f"\nEne: {response}\n")

    finally:
        await graphiti.close()
        log.info("🔒 Graphiti connection closed")


if __name__ == "__main__":
    asyncio.run(main())
