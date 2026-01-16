import sys
import os
import math
import time
import json
from pathlib import Path
import queue

import numpy as np
import sounddevice as sd

from PySide6.QtWidgets import QApplication
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QCoreApplication
from PySide6.QtGui import QSurfaceFormat

from OpenGL.GL import *
import live2d.v3 as live2d

from kokoro import KPipeline

from llama_cpp import Llama

# ---------------- Configuration ----------------
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
FPS = 60

LIP_SYNC_SENSITIVITY = 0.4 
LIP_SYNC_SMOOTHING = 0.35
LIP_SYNC_THRESHOLD = 0.005

# TTS
TTS_VOICE = "af_aoede"
SAMPLE_RATE = 22050
FRAME_SIZE = 441

# Adaptive audio tuning (change these to taste)
AUDIO_NOISE_FLOOR = 1e-4    # below this RMS we treat as silence
AUDIO_VOLUME_GAIN = 6.0     # how aggressively RMS maps to mouth movement
AUDIO_COMPRESSION = 0.8     # <1 compresses peaks (reduces clipping)

MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=32768,
    n_gpu_layers=-1,
    n_threads=8,
    verbose=False,
)

SYSTEM_PROMPT = Path("system_prompt.txt").read_text(
    encoding="utf-8"
).strip()


def llm_stream(user_msg: str):
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}
Your creator is teo.
<|im_end|>
<|im_start|>user
{user_msg}
<|im_end|>
<|im_start|>assistant
"""

    start = time.perf_counter()
    first_token = None
    token_count = 0

    spoken_chunks = []

    for chunk in llm(
        prompt,
        max_tokens=256,
        temperature=0.5,
        top_p=0.9,
        stop=["<|im_end|>"],
        stream=True,
    ):
        text = chunk["choices"][0]["text"]
        if not text:
            continue

        if first_token is None:
            first_token = time.perf_counter()

        token_count += 1
        spoken_chunks.append(text)

        yield text  

    end = time.perf_counter()

    if first_token:
        ttft = first_token - start
        tps = token_count / (end - first_token)
        print(
            f"\n⏱ TTFT: {ttft:.3f}s | 🧮 Tokens: {token_count} | ⚡ {tps:.1f} tok/s | ⌛ Total: {end - start:.2f}s\n"
        )

    llm_stream.last_spoken_text = "".join(spoken_chunks)

class LLMWorker(QThread):
    text_chunk = Signal(str)
    finished = Signal()

    def __init__(self, user_text: str):
        super().__init__()
        self.user_text = user_text
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        buffer = ""
        for token in llm_stream(self.user_text):
            if not self._running:
                break

            buffer += token
            if any(p in buffer for p in ".!?"):
                self.text_chunk.emit(buffer.strip())
                buffer = ""

        if buffer.strip():
            self.text_chunk.emit(buffer.strip())

        self.finished.emit()

# ---------------- Helper ----------------

def get_fixed_model_path(original_path: Path):
    fixed_path = original_path.with_stem(original_path.stem + "_fixed")
    if fixed_path.exists():
        return fixed_path

    with open(original_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.pop("DefaultExpression", None)
    if "FileReferences" in data:
        data["FileReferences"].pop("DefaultExpression", None)

    with open(fixed_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return fixed_path

# ---------------- Audio Worker ----------------
class AudioWorker(QThread):
    # emits *raw* RMS (not yet scaled to mouth) so UI code can adapt
    volume_update = Signal(float)

    def __init__(self):
        super().__init__()
        self.running = True
        self.audio_queue = queue.Queue()

    def add_audio(self, audio: np.ndarray):
        # ensure float32 numpy array
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        self.audio_queue.put(audio)

    def run(self):
        try:
            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SIZE
            ) as stream:

                while self.running:
                    try:
                        audio = self.audio_queue.get(timeout=0.05)

                        # write in blocks and compute RMS per-block
                        for i in range(0, len(audio), FRAME_SIZE):
                            if not self.running:
                                break

                            frame = audio[i:i + FRAME_SIZE]
                            if len(frame) == 0:
                                continue

                            # remove any DC offset before RMS
                            frame = frame - np.mean(frame)

                            stream.write(frame.reshape(-1, 1))

                            # RMS (raw, unscaled)
                            rms = float(np.sqrt(np.mean(frame ** 2)))

                            # emit raw RMS
                            self.volume_update.emit(rms)

                    except queue.Empty:
                        # emit 0 so UI can decay/mute
                        self.volume_update.emit(0.0)

        except Exception as e:
            # print a single error message so user knows something went wrong
            print(f"[AudioWorker] Error: {e}")

    def stop(self):
        self.running = False
        self.wait()

# ---------------- TTS Generator ----------------
class TTSGenerator(QThread):
    audio_chunk = Signal(np.ndarray)
    finished_tts = Signal()

    def __init__(self, pipeline: KPipeline, text: str):
        super().__init__()
        self.pipeline = pipeline
        self.text = text

    def run(self):
        try:
            for r in self.pipeline(self.text, voice=TTS_VOICE):
                audio = (
                    r.output.audio
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                # Remove DC offset and safety-clip to [-1, 1]
                audio = audio - np.mean(audio)
                np.clip(audio, -1.0, 1.0, out=audio)

                self.audio_chunk.emit(audio)

        except Exception as e:
            print(f"[TTS] Error: {e}")

        self.finished_tts.emit()

# ---------------- CLI Input ----------------
class InputThread(QThread):
    text_received = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        print("\nKokoro — type text and press Enter. Commands: /help /quit\n")
        while self.running:
            try:
                text = input()
                if not text:
                    continue

                if text.strip() in ("/quit", "/exit"):
                    QCoreApplication.quit()
                    break

                self.text_received.emit(text)
            except EOFError:
                break

# ---------------- Live2D Widget ----------------
class Live2DWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()

        self.model = None
        self.start_time = time.time()

        self._tts_start_time = None

        # mouth values
        self.mouth_raw = 0.0
        self.mouth_value = 0.0

        # adaptive peak for safety (prevents mouth from getting "stuck open")
        self._peak = 1e-6

        self.audio_worker = AudioWorker()
        self.audio_worker.volume_update.connect(self.on_volume)

        self.pipeline = None
        self.tts_queue = queue.Queue()
        self.tts_busy = False

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.drag_position = None
        self.setMouseTracking(True)

        fmt = QSurfaceFormat()
        fmt.setVersion(2, 1)
        fmt.setAlphaBufferSize(8)
        self.setFormat(fmt)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // FPS)

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def enqueue_tts(self, text):
        if hasattr(self, "llm_worker") and self.llm_worker.isRunning():
            self.llm_worker.stop()
            self.llm_worker.wait()

        self.llm_worker = LLMWorker(text)
        self.llm_worker.text_chunk.connect(self.tts_queue.put)
        self.llm_worker.finished.connect(self.process_queue)
        self.llm_worker.start()

    def process_queue(self):
        if self.tts_busy or self.tts_queue.empty():
            return

        self.tts_busy = True
        text = self.tts_queue.get()

        self.current_tts = TTSGenerator(self.pipeline, text)
        self.current_tts.audio_chunk.connect(self.audio_worker.add_audio)
        self.current_tts.finished_tts.connect(self.on_tts_done)

        self._tts_start_time = time.time()
        self.current_tts.start()

    def on_tts_done(self):
        if self._tts_start_time is not None:
            elapsed = time.time() - self._tts_start_time

            print(f"\ninference: {elapsed:.2f}s\n")

        self._tts_start_time = None
        self.tts_busy = False
        self.process_queue()

    def initializeGL(self):
        live2d.glInit()
        glClearColor(0, 0, 0, 0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        path = get_fixed_model_path(Path("tororo_vts/tororo.model3.json"))

        self.model = live2d.LAppModel()
        self.model.LoadModelJson(str(path))
        self.model.Resize(self.width(), self.height())

        self.audio_worker.start()

    def on_volume(self, v_raw):
        # v_raw is the RMS computed per-frame in AudioWorker

        # floor small noise
        v = max(0.0, v_raw - AUDIO_NOISE_FLOOR)

        # simple compressor / gain
        v = v * AUDIO_VOLUME_GAIN
        v = v ** AUDIO_COMPRESSION

        # track a slowly-decaying peak to avoid permanent over-scaling
        self._peak = max(self._peak * 0.995, max(v, 1e-6))

        # normalize by peak so loud clips don't permanently dominate
        v = v / (self._peak + 1e-6)

        # final sensitivity multiplier
        v = min(max(v * LIP_SYNC_SENSITIVITY, 0.0), 1.0)

        # apply slight hold so very short drops don't fully close mouth
        self.mouth_raw = max(self.mouth_raw * 0.85, v)

    def paintGL(self):
        live2d.clearBuffer()
        if not self.model:
            return

        t = time.time() - self.start_time
        self.model.SetParameterValue("PARAM_BREATH", (math.sin(t * 2) + 1) / 2)

        if self.mouth_raw > LIP_SYNC_THRESHOLD:
            target = min(self.mouth_raw, 1.0)
        else:
            target = 0.0

        self.mouth_value += (target - self.mouth_value) * LIP_SYNC_SMOOTHING

        for p in ("PARAM_MOUTH_OPEN_Y", "ParamMouthOpenY"):
            self.model.SetParameterValue(p, self.mouth_value)

        self.model.Update()
        self.model.Draw()

    def closeEvent(self, event):
        if hasattr(self, "llm_worker") and self.llm_worker.isRunning():
            self.llm_worker.stop()
            self.llm_worker.wait()

        if hasattr(self, "current_tts") and self.current_tts.isRunning():
            self.current_tts.quit()
            self.current_tts.wait()

        self.audio_worker.stop()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_position is not None:
            self.move(
                event.globalPosition().toPoint()
                - self.drag_position
            )
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = None
            event.accept()


# ---------------- Main ----------------
def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    app = QApplication(sys.argv)

    print("[Kokoro] Ready — enter text to speak (type /help in console for commands)")
    pipeline = KPipeline(lang_code="a", device="cpu")

    live2d.init()

    widget = Live2DWidget()
    widget.set_pipeline(pipeline)
    widget.show()

    input_thread = InputThread()
    input_thread.text_received.connect(widget.enqueue_tts)
    input_thread.start()

    try:
        sys.exit(app.exec())
    finally:
        input_thread.stop()
        input_thread.wait()
        live2d.dispose()

if __name__ == "__main__":
    main()
