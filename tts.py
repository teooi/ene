import time
import torch
import numpy as np
import sounddevice as sd
from kokoro import KPipeline
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# -----------------------------
# Device selection
# -----------------------------
# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

device = "cpu"

print(f"Using device: {device}")

# -----------------------------
# Initialize pipeline ONCE
# -----------------------------
print("Initializing Kokoro pipeline...")
pipeline = KPipeline(lang_code="a", device=device)
print("Pipeline ready.\n")

SAMPLE_RATE = 22050

# -----------------------------
# Continuous TTS loop
# -----------------------------
while True:
    try:
        text = input("\nEnter text (or empty to quit): ").strip()
        if not text:
            print("Exiting.")
            break

        print("Starting inference timer...")
        t0 = time.perf_counter()

        gen = pipeline(text, voice="af_aoede")

        audio_chunks = []
        for r in gen:
            audio_chunks.append(
                r.output.audio.detach().cpu().numpy()
            )

        audio = np.concatenate(audio_chunks)

        t1 = time.perf_counter()
        latency = t1 - t0

        print(f"Inference complete.")
        print(f"Latency: {latency:.3f} sec")
        print(f"Audio duration: {len(audio) / SAMPLE_RATE:.2f} sec")

        print("Playing audio...")
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        break
