import sys
from neo4j import GraphDatabase
from llama_cpp import Llama
from pprint import pprint

# -----------------------------
# CONFIG
# -----------------------------

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

GGUF_MODEL_PATH = "/Users/teoi/Documents/ene/models/Meta-Llama-3-8B-Instruct.Q4_1.gguf"
CTX_SIZE = 4096

# -----------------------------
# INIT NEO4J
# -----------------------------

print("🔌 Connecting to Neo4j...")
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)
print("✅ Neo4j connected\n")

# -----------------------------
# INIT GGUF
# -----------------------------

print("🧠 Loading GGUF model...")
llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=8,
    n_gpu_layers=-1,   
    verbose=False,
)
print("✅ GGUF loaded\n")

# -----------------------------
# GRAPH RETRIEVAL
# -----------------------------

def retrieve_memory(user_query: str, limit: int = 5):
    print("\n🔍 MEMORY RETRIEVAL")
    print("User query:", user_query)

    cypher = """
    CALL db.index.fulltext.queryRelationships(
        "edge_name_and_fact",
        $query,
        { limit: $limit }
    )
    YIELD relationship AS rel, score
    MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)
    RETURN
        n.name AS source,
        e.fact AS fact,
        m.name AS target,
        score
    ORDER BY score DESC
    """

    params = {
        "query": user_query,
        "limit": limit
    }

    print("\n🧾 Cypher Query:")
    print(cypher.strip())
    print("Params:", params)

    with driver.session() as session:
        records = list(session.run(cypher, params))

    print("\n📦 Raw Neo4j Records:")
    for r in records:
        print(dict(r))

    memories = []
    for r in records:
        memories.append({
            "source": r["source"],
            "fact": r["fact"],
            "target": r["target"],
            "score": r["score"],
        })

    print("\n🧠 Parsed Memories:")
    for m in memories:
        print(m)

    return memories

# -----------------------------
# PROMPT BUILDING
# -----------------------------

def build_prompt(user_input: str, memories: list[dict]) -> str:
    print("\n🧾 BUILDING PROMPT")

    if memories:
        context = "\n".join(
            f"- {m['fact']}" for m in memories
        )
    else:
        context = "None."

    prompt = f"""
You are Ene.

You have access to long-term memory retrieved from a knowledge graph.
If MEMORY contains relevant facts, you MUST use them.
If MEMORY is empty, say you do not know.
Do NOT invent facts.

MEMORY:
{context}

USER:
{user_input}

ENE:
"""

    print("\n📤 PROMPT SENT TO MODEL:")
    print("=" * 80)
    print(prompt.strip())
    print("=" * 80)

    return prompt

# -----------------------------
# CHAT LOOP
# -----------------------------

def chat():
    print("🧠 Ene memory chat (Ctrl+C to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            memories = retrieve_memory(user_input)
            prompt = build_prompt(user_input, memories)

            print("\n🤖 GENERATING RESPONSE...\n")

            output = llm(
                prompt,
                max_tokens=512,
                temperature=0.4,
                stop=["You:"]
            )

            response = output["choices"][0]["text"].strip()

            print("\n🧠 RAW MODEL OUTPUT:")
            pprint(output)

            print("\nEne:", response, "\n")

        except KeyboardInterrupt:
            print("\n👋 bye.")
            break

# -----------------------------
# ENTRY
# -----------------------------

if __name__ == "__main__":
    chat()
