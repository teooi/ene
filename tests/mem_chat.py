import asyncio
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Dict

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

MODEL_PATH = "/Users/teoi/Documents/ene/models/Meta-Llama-3-8B-Instruct.Q4_1.gguf"
SUMMARY_MODEL_PATH = "/Users/teoi/Documents/ene/models/gemma-2-2b-it-Q4_K_M.gguf"
SYSTEM_PROMPT_PATH = "/Users/teoi/Documents/ene/system_prompt.txt"

MAX_MEMORY_FACTS = 3
MAX_SUMMARY_TOKENS = 250  # Increased for detailed state tracking
MAX_SUMMARY_WORDS = 200   # Up from 40
HARD_RESET_LIMIT = 10     # Hard limit for auto-reset
SUMMARY_LOG_PATH = "/Users/teoi/Documents/ene/short_term_memory_logs.txt"
SUMMARY_STATE_PATH = "/Users/teoi/Documents/ene/current_summary.txt"

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
        log.info(f"Initializing main Llama model: {MODEL_PATH}")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=15000,
            n_threads=8,
            n_gpu_layers=-1,
            verbose=False,
        )
        log.info("Main Llama model loaded successfully")
        return llm
    except Exception as e:
        log.error(f"Failed to initialize main LLM: {e}")
        traceback.print_exc()
        return None

def init_summary_llm() -> Optional[Llama]:
    try:
        log.info(f"Initializing summary model: {SUMMARY_MODEL_PATH}")
        llm = Llama(
            model_path=SUMMARY_MODEL_PATH,
            n_ctx=4096,  # Increased from 2048
            n_threads=4,
            n_gpu_layers=-1,
            verbose=False,
            n_batch=512,
        )
        log.info("Summary model loaded successfully")
        return llm
    except Exception as e:
        log.error(f"Failed to initialize summary LLM: {e}")
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
        return "You are a helpful AI assistant."

# ======================================================
# LLM-BASED CONTEXT GATING
# ======================================================

def ask_llm_about_context(
    llm: Llama,
    user_input: str,
    current_summary: Optional[str],
) -> bool:
    """
    Ask the LLM if we should use long-term memories for this query.
    Returns True if long-term context needed, False if current context is sufficient.
    """
    if not current_summary:
        return True  # No current context, always retrieve
    
    prompt = f"""Current conversation context: {current_summary}

User's new message: {user_input}

Question: Does this message require retrieving old memories/facts about the user, or can it be answered using just the current conversation context?

Answer with ONLY "RETRIEVE" or "CONTEXT":
- RETRIEVE: if the user is asking about past facts, changing topics, or needs historical information
- CONTEXT: if continuing the current activity/conversation and past facts would be irrelevant or distracting

Answer:"""

    try:
        output = llm(
            prompt,
            max_tokens=10,
            temperature=0.1,
            stop=["\n", ".", ","],
        )
        
        decision = output["choices"][0]["text"].strip().upper()
        should_retrieve = "RETRIEVE" in decision
        
        log.info(f"🤔 Context decision: {'RETRIEVE memories' if should_retrieve else 'USE current context only'}")
        return should_retrieve
        
    except Exception as e:
        log.warning(f"Context gating failed: {e}, defaulting to RETRIEVE")
        return True  # Default to retrieving on error

# ======================================================
# LLM-BASED RESET DECISION
# ======================================================

def should_reset_summary(
    llm: Llama,
    current_summary: str,
    turns_since_reset: int,
) -> bool:
    """Ask LLM if the summary should be reset."""
    
    # Hard limit
    if turns_since_reset >= HARD_RESET_LIMIT:
        log.info(f"🔄 Hard limit: forcing reset after {HARD_RESET_LIMIT} turns")
        return True
    
    # Early turns: don't reset
    if turns_since_reset < 3:
        return False
    
    # Ask LLM if context is getting stale
    prompt = f"""Current conversation summary: {current_summary}

This summary has been accumulating for {turns_since_reset} conversation turns.

Question: Is this summary still useful and focused, or has it become stale/repetitive/stuck on an old topic?

Answer with ONLY "KEEP" or "RESET":
- KEEP: if the summary is still relevant and useful
- RESET: if it's getting stale, repetitive, or stuck on an old topic

Answer:"""

    try:
        output = llm(
            prompt,
            max_tokens=10,
            temperature=0.1,
            stop=["\n", ".", ","],
        )
        
        decision = output["choices"][0]["text"].strip().upper()
        should_reset = "RESET" in decision
        
        if should_reset:
            log.info(f"🤔 LLM decided: summary is stale after {turns_since_reset} turns, resetting")
        
        return should_reset
        
    except Exception as e:
        log.warning(f"Reset check failed: {e}, keeping summary")
        return False

# ======================================================
# PROMPT BUILDERS
# ======================================================

def build_response_prompt(
    system_prompt: str,
    user_input: str,
    memories: List[str],
    short_term_summary: Optional[str] = None,
) -> str:
    """Build prompt for immediate response to user."""
    parts = [system_prompt]
    
    if memories:
        memory_block = "Long-term memory:\n" + "\n".join(f"- {m}" for m in memories)
        parts.append(memory_block)
    
    if short_term_summary:
        parts.append(f"Recent conversation context:\n{short_term_summary}")
    
    parts.append(f"User: {user_input}\nEne:")
    
    return "\n\n".join(parts)

def build_summary_prompt(
    conversation_history: List[Dict[str, str]],
    previous_summary: Optional[str] = None,
    should_forget: bool = False,
) -> str:
    """Build prompt for generating conversation summary."""
    
    # Use last 3 turns for richer context
    recent_turns = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
    history_text = "\n".join(
        f"User: {turn['user']}\nEne: {turn['ene']}" 
        for turn in recent_turns
    )
    
    if should_forget:
        # Even when forgetting, be somewhat detailed
        prompt = f"""Recent conversation:
{history_text}

Summarize what's happening in 1-2 sentences (max 40 words). You can be vague about details.

Summary:"""
    else:
        # Rich, detailed summary
        prompt = f"""Recent conversation:
{history_text}

Provide a detailed summary of the current situation in 2-3 sentences (max 200 words). If there's an ongoing activity, describe its current state. If it's a discussion, capture the key points and where things stand.

Summary:"""
    
    return prompt

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
    """
    Extract a single clean semantic fact for LLM consumption.
    Priority: summary > content > fact
    """
    if getattr(node, "summary", None):
        return node.summary.strip()

    if getattr(node, "content", None):
        return node.content.strip()

    if getattr(node, "fact", None):
        return str(node.fact).strip()

    return None

async def retrieve_memories(
    graphiti: Graphiti,
    user_input: str,
) -> List[str]:
    """Retrieve relevant memories from graph."""
    
    # Special case for birthday queries
    if any(k in user_input.lower() for k in ("birthday", "born")):
        query = "Teo born birthday date"
    else:
        query = user_input

    log.info(f"🔍 Querying memory graph for: {query!r}")
    
    search_results = await graphiti._search(query, config=NODE_HYBRID_SEARCH_RRF)
    results = search_results.nodes

    if results:
        log.info(f"🧠 Retrieved {len(results)} memory nodes from graph")
    else:
        log.info("🧠 No memory nodes returned")

    # Extract facts without filtering - trust the search ranking
    memories: List[str] = []
    for r in results[:MAX_MEMORY_FACTS]:  # Just take top N
        fact = node_to_memory_fact(r)
        if fact and len(fact.split()) >= 3:  # Only basic length check
            memories.append(fact)

    if memories:
        log.info(f"🧠 Passing {len(memories)} memory facts to LLM:")
        for i, m in enumerate(memories, 1):
            log.info(f"   [{i}] {m}")
    else:
        log.info("🧠 No usable memory facts found")

    return memories

# ======================================================
# RESPONSE GENERATION
# ======================================================

def generate_immediate_response(
    llm: Llama,
    prompt: str,
) -> str:
    """Generate the immediate response to user (Part 1)."""
    log.info("🎯 Generating immediate response...")
    
    output = llm(
        prompt,
        max_tokens=256, 
        temperature=0.8,  
        stop=["User:", "\nUser:", "\n\n", "```", "```python", '"""'],  
    )

    response = output["choices"][0]["text"].strip()

    if response.startswith("Ene:"):
        response = response[4:].strip()

    # Clean up response
    import re
    response = re.sub(r'```[\s\S]*?```', '', response)
    response = re.sub(r'```.*', '', response)
    response = re.sub(r'\([^)]*\)', '', response)
    response = re.sub(r'\*[^*]*\*', '', response)
    response = response.replace('"""', '')
    response = ' '.join(response.split())

    return response

def log_summary_change(
    turn_number: int,
    user_input: str,
    old_summary: Optional[str],
    new_summary: str,
) -> None:
    """Log summary changes to file for debugging."""
    try:
        timestamp = Path(SUMMARY_LOG_PATH).exists()
        mode = 'a' if timestamp else 'w'
        
        with open(SUMMARY_LOG_PATH, mode, encoding='utf-8') as f:
            from datetime import datetime
            f.write(f"\n{'='*60}\n")
            f.write(f"Turn {turn_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
            f.write(f"User: {user_input}\n\n")
            f.write(f"Old Summary: {old_summary or '(None)'}\n")
            f.write(f"New Summary: {new_summary}\n")
    except Exception as e:
        log.warning(f"Failed to log summary: {e}")

def generate_summary(
    summary_llm: Llama,
    summary_prompt: str,
    should_forget: bool = False,
) -> str:
    """Generate conversation summary using lightweight model."""
    log.info(f"📝 Generating conversation summary{' (forgetting mode)' if should_forget else ''}...")
    
    try:
        output = summary_llm(
            summary_prompt,
            max_tokens=MAX_SUMMARY_TOKENS,
            temperature=0.3,
            stop=["\n\n\n", "User:", "Ene:", "Recent conversation:"],
            repeat_penalty=1.1,
        )

        summary = output["choices"][0]["text"].strip()
        
        # Clean up artifacts
        summary = summary.replace("Summary:", "").strip()
        summary = summary.replace("The user is", "User").strip()
        summary = summary.replace("They are", "").strip()
        
        # Take first few sentences (up to limit)
        words = summary.split()
        word_limit = 40 if should_forget else MAX_SUMMARY_WORDS
        
        if len(words) > word_limit:
            # Find sentence boundary near limit
            truncated = ' '.join(words[:word_limit])
            # Try to end at a sentence
            if '.' in truncated:
                last_period = truncated.rfind('.')
                summary = truncated[:last_period + 1]
            else:
                summary = truncated
            log.warning(f"⚠️ Summary truncated to ~{word_limit} words")
        
        log.info(f"📝 Summary ({len(summary.split())} words): {summary}")
        return summary
        
    except Exception as e:
        log.error(f"Summary generation failed: {e}")
        return ""

# ======================================================
# CHAT LOOP
# ======================================================

async def chat_loop(llm: Llama, summary_llm: Llama, graphiti: Graphiti) -> None:
    system_prompt = load_system_prompt()
    
    # Track conversation state
    conversation_history: List[Dict[str, str]] = []
    current_summary: Optional[str] = None
    turns_since_reset = 0
    turn_number = 0

    print("\n🧠 Ene Memory Chat (type 'exit' or 'reset')\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        
        # Manual reset command
        if user_input.lower() == "reset":
            conversation_history.clear()
            old_summary = current_summary
            current_summary = None
            turns_since_reset = 0
            turn_number += 1
            log_summary_change(turn_number, "MANUAL RESET", old_summary, "(Reset)")
            print("🔄 Short-term memory reset\n")
            continue

        turn_number += 1

        # LLM-based memory retrieval gating
        should_retrieve = ask_llm_about_context(llm, user_input, current_summary)
        
        if should_retrieve:
            memories = await retrieve_memories(graphiti, user_input)
        else:
            memories = []
        
        response_prompt = build_response_prompt(
            system_prompt, 
            user_input, 
            memories, 
            current_summary
        )
        
        immediate_response = generate_immediate_response(llm, response_prompt)
        print(f"\nEne: {immediate_response}\n")

        # Add to conversation history
        conversation_history.append({
            "user": user_input,
            "ene": immediate_response,
        })
        turns_since_reset += 1

        # LLM-based adaptive reset
        if current_summary and should_reset_summary(llm, current_summary, turns_since_reset):
            old_summary = current_summary
            current_summary = None
            turns_since_reset = 0
            conversation_history = conversation_history[-1:]
            log_summary_change(turn_number, user_input, old_summary, "(LLM-triggered reset)")
            continue

        # Part 2: Generate summary using lightweight model
        # Randomly forget details (like humans) - reduced probability
        import random
        should_forget = random.random() < 0.15
        
        old_summary = current_summary
        summary_prompt = build_summary_prompt(conversation_history, current_summary, should_forget)
        current_summary = generate_summary(summary_llm, summary_prompt, should_forget)
        
        # If summary generation failed, clear it
        if not current_summary:
            current_summary = None
            log.warning("⚠️ Summary cleared due to generation error")
        
        # Log the summary change
        log_summary_change(turn_number, user_input, old_summary, current_summary or "(Empty)")

# ======================================================
# MAIN
# ======================================================

async def main():
    llm = init_llm()
    if not llm:
        return
    
    summary_llm = init_summary_llm()
    if not summary_llm:
        return

    log.info("🧱 Initializing Graphiti (chat mode)")
    graphiti = await init_graphiti()

    try:
        await chat_loop(llm, summary_llm, graphiti)
    finally:
        await graphiti.close()
        log.info("🔒 Graphiti connection closed")

if __name__ == "__main__":
    asyncio.run(main())