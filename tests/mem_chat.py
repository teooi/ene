import asyncio
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Dict
from collections import deque

from llama_cpp import Llama

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

# ======================================================
# CONFIG
# ======================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_LLM_MODEL = "llama3.1:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

MODEL_PATH = "/Users/teoi/Documents/ene/models/ENE-Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"
SYSTEM_PROMPT_PATH = "/Users/teoi/Documents/ene/system_prompt.txt"

# Thresholds & sizes
MIN_MEMORY_WORDS = 3
MAX_MEMORY_FACTS = 3
MAX_CONTEXT_TURNS = 10

# ======================================================
# LOGGING
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ene.chat")

# ======================================================
# LLM INIT
# ======================================================

def init_llm() -> Optional[Llama]:
    try:
        log.info(f"Initializing Llama model: {MODEL_PATH}")
        # chat_format="llama-3" is usually auto-detected, but good to ensure
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_threads=8,
            n_gpu_layers=-1,
            verbose=False,
        )
        log.info("Llama model loaded successfully")
        return llm
    except Exception as e:
        log.error(f"Failed to initialize LLM: {e}")
        traceback.print_exc()
        return None

# ======================================================
# SYSTEM PROMPT
# ======================================================

def load_system_prompt() -> str:
    try:
        return Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        log.warning("system_prompt.txt not found, using fallback prompt")
        return "You are Ene. You are sarcastic, blunt, and extremely concise."

# ======================================================
# PROMPT BUILDERS (UPDATED)
# ======================================================

def build_chat_messages(
    system_prompt: str,
    user_input: str,
    memories: List[str],
    recent_context: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Builds a structured list of messages for Llama 3's chat template.
    """
    messages = []

    # 1. System Prompt + Memories
    # We inject memories into the system prompt so they feel like "background knowledge"
    full_system_msg = system_prompt
    if memories:
        full_system_msg += "\n\nRELEVANT MEMORY FACTS:\n" + "\n".join(f"- {m}" for m in memories)
    
    messages.append({"role": "system", "content": full_system_msg})

    # 2. Recent Conversation History
    for turn in recent_context:
        messages.append({"role": "user", "content": turn['user']})
        messages.append({"role": "assistant", "content": turn['ene']})

    # 3. Current User Input
    messages.append({"role": "user", "content": user_input})

    return messages

# ======================================================
# GRAPHITI INIT
# ======================================================

async def init_graphiti() -> Graphiti:
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

    return Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=reranker,
    )

# ======================================================
# MEMORY SEARCH
# ======================================================

def node_to_memory_fact(node) -> Optional[str]:
    if getattr(node, "summary", None):
        return node.summary.strip()
    if getattr(node, "content", None):
        return node.content.strip()
    if getattr(node, "fact", None):
        return str(node.fact).strip()
    return None

async def retrieve_memories(graphiti: Graphiti, user_input: str) -> List[str]:
    if any(k in user_input.lower() for k in ("birthday", "born")):
        query = "Teo born birthday date"
    else:
        query = user_input

    log.info(f"🔍 Querying memory graph for: {query!r}")
    
    search_results = await graphiti._search(query, config=NODE_HYBRID_SEARCH_RRF)
    results = getattr(search_results, "nodes", None) or []

    facts: List[str] = []
    for n in results:
        fact = node_to_memory_fact(n)
        if not fact or len(fact.split()) < MIN_MEMORY_WORDS:
            continue
        facts.append(fact)

    kept = facts[:MAX_MEMORY_FACTS]
    if kept:
        log.info(f"🧠 Passing {len(kept)} memory facts to LLM:")
        for i, m in enumerate(kept, 1):
            log.info(f"   [{i}] {m}")
    else:
        log.info("🧠 No usable memory facts found")

    return kept

# ======================================================
# RESPONSE GENERATION (UPDATED)
# ======================================================

def generate_response(
    llm: Llama,
    messages: List[Dict[str, str]],
) -> str:
    """Generate response using create_chat_completion."""
    log.info("🎯 Generating response...")
    
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,        
        temperature=1.0,     
        repeat_penalty=1.2,   
        top_p=0.9,
    )

    response = output["choices"][0]["message"]["content"].strip()

    # Clean up standard junk if it appears
    import re
    response = re.sub(r'```[\s\S]*?```', '', response)
    response = re.sub(r'\*[^*]*\*', '', response) # Remove actions *laughs*
    
    return response

# ======================================================
# CHAT LOOP (UPDATED)
# ======================================================

async def chat_loop(llm: Llama, graphiti: Graphiti) -> None:
    system_prompt = load_system_prompt()
    recent_context = deque(maxlen=MAX_CONTEXT_TURNS)

    print("\n🧠 Ene Memory Chat (type 'exit' to quit, 'reset' to clear context)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            break
        
        if user_input.lower() == "reset":
            recent_context.clear()
            print("🔄 Context cleared\n")
            continue

        # Retrieve memories
        memories = await retrieve_memories(graphiti, user_input)

        # Build STRUCTURED messages instead of raw text string
        messages = build_chat_messages(
            system_prompt, 
            user_input, 
            memories, 
            list(recent_context)
        )
        
        # Generate response
        response = generate_response(llm, messages)
        print(f"\nEne: {response}\n")

        recent_context.append({
            "user": user_input,
            "ene": response,
        })
        
        log.info(f"📝 Context window: {len(recent_context)}/{MAX_CONTEXT_TURNS} turns")

# ======================================================
# MAIN
# ======================================================

async def main():
    llm = init_llm()
    if not llm:
        log.error("Main LLM failed to initialize; exiting.")
        return

    log.info("🧱 Initializing Graphiti")
    graphiti = await init_graphiti()

    try:
        await chat_loop(llm, graphiti)
    finally:
        await graphiti.close()
        log.info("🔒 Graphiti connection closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted, exiting.")