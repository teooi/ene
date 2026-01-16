# logger.py
from config import DEBUG

def log(msg: str, level: str = "DEBUG"):
    if DEBUG or level != "DEBUG":
        print(f"[{level}] {msg}", flush=True)
