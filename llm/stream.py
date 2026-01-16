import time
import traceback
from logger import log

def llm_stream(llm, system_prompt, user_msg):
    if not llm:
        yield "LLM not loaded."
        return

    prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_msg}
<|im_end|>
<|im_start|>assistant
"""

    start = time.perf_counter()
    first_token = None
    token_count = 0

    try:
        for chunk in llm(
            prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stop=["<|im_end|>"],
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            if not text:
                continue

            if first_token is None:
                first_token = time.perf_counter()
                log(f"First token: {first_token - start:.3f}s")

            token_count += 1
            yield text

    except Exception as e:
        log(f"LLM stream error: {e}", "ERROR")
        traceback.print_exc()
        yield " ...error."
