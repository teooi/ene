from pathlib import Path
import traceback
from llama_cpp import Llama
from typing import Optional
from config import MODEL_PATH
from logger import log

def init_llm() -> Optional[Llama]:
    try:
        log(f"Initializing Llama model: {MODEL_PATH}")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_gpu_layers=-1,
            n_threads=8,
            verbose=False,
        )
        log("Llama model loaded successfully")
        return llm
    except Exception as e:
        log(f"Failed to initialize LLM: {e}", "ERROR")
        traceback.print_exc()
        return None

def load_system_prompt() -> str:
    try:
        return Path("system_prompt.txt").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        log("system_prompt.txt not found", "WARNING")
        return "You are a helpful AI assistant."
