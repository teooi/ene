import time
from llama_cpp import Llama
from pathlib import Path

MODEL_PATH = "models/qwen2.5-3b-instruct-q4_k_m.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=32768,
    n_gpu_layers=-1,
    n_threads=8,
    verbose=False,
)

SYSTEM_PROMPT = Path("system_prompt.txt").read_text(encoding="utf-8").strip()

def chat_once(user_msg: str):
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
{user_msg}
<|im_end|>
<|im_start|>assistant
"""

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    for chunk in llm(
        prompt,
        max_tokens=256,
        temperature=0.5,
        top_p=0.9,
        stop=["<|im_end|>"],
        stream=True,
    ):
        text = chunk["choices"][0]["text"]
        if text:
            if first_token_time is None:
                first_token_time = time.perf_counter()

            token_count += 1
            print(text, end="", flush=True)

    end_time = time.perf_counter()

    total_time = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else 0
    tps = token_count / (end_time - first_token_time) if first_token_time else 0

    print("\n")
    print(f"⏱ TTFT: {ttft:.3f}s | 🧮 Tokens: {token_count} | ⚡ {tps:.1f} tok/s | ⌛ Total: {total_time:.2f}s")
    print()

if __name__ == "__main__":
    print("Qwen2.5 local chat test with timing (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        print("Assistant: ", end="", flush=True)
        chat_once(user_input)