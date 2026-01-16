import sys
import os
import math
import time
import json
from pathlib import Path
import queue
import signal
import traceback
from typing import Optional

import numpy as np
import sounddevice as sd

from PySide6.QtWidgets import QApplication
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QCoreApplication, QMutex, QMutexLocker
from PySide6.QtGui import QSurfaceFormat, QGuiApplication

from OpenGL.GL import *
import live2d.v3 as live2d

from kokoro import KPipeline
from llama_cpp import Llama

import librosa

# ============================================================
# CONFIGURATION
# ============================================================
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
FPS = 60

# Lip sync parameters
LIP_SYNC_SENSITIVITY = 0.7
LIP_SYNC_SMOOTHING = 0.35
LIP_SYNC_THRESHOLD = 0.005

# TTS settings
TTS_VOICE = "af_heart"
SAMPLE_RATE = 22050
FRAME_SIZE = 441

# Audio processing
AUDIO_NOISE_FLOOR = 1e-4
AUDIO_VOLUME_GAIN = 6.0
AUDIO_COMPRESSION = 0.8

# Model paths
MODEL_PATH = "models/Meta-Llama-3-8B-Instruct.Q4_1.gguf"
LIVE2D_MODEL_PATH = "live2d/l2d/L2DZeroVS.model3.json"

# Debug mode
DEBUG = True

def log(msg: str, level: str = "DEBUG"):
    """Thread-safe logging"""
    if DEBUG or level != "DEBUG":
        print(f"[{level}] {msg}", flush=True)

# ============================================================
# LLM INITIALIZATION
# ============================================================
def init_llm() -> Optional[Llama]:
    """Initialize LLM with error handling"""
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

llm = init_llm()

def load_system_prompt() -> str:
    """Load system prompt with fallback"""
    try:
        prompt = Path("system_prompt.txt").read_text(encoding="utf-8").strip()
        log(f"System prompt loaded ({len(prompt)} chars)")
        return prompt
    except FileNotFoundError:
        log("system_prompt.txt not found, using default", "WARNING")
        return "You are a helpful AI assistant named Ene."
    except Exception as e:
        log(f"Error loading system prompt: {e}", "ERROR")
        return "You are a helpful AI assistant."

SYSTEM_PROMPT = load_system_prompt()

# ============================================================
# LLM STREAMING
# ============================================================
def llm_stream(user_msg: str):
    """Stream LLM response with comprehensive error handling"""
    if not llm:
        log("LLM not initialized, cannot stream", "ERROR")
        yield "I'm sorry, my language model isn't loaded."
        return
    
    log(f"llm_stream called: '{user_msg[:50]}...'")
    
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}
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
            spoken_chunks.append(text)
            
            if DEBUG:
                print(f"[Token {token_count}] '{text}'", end='', flush=True)

            yield text

    except Exception as e:
        log(f"LLM stream error: {e}", "ERROR")
        traceback.print_exc()
        yield " ...I encountered an error."

    end = time.perf_counter()

    if first_token and token_count > 0:
        ttft = first_token - start
        tps = token_count / (end - first_token) if end > first_token else 0
        log(f"TTFT: {ttft:.3f}s | Tokens: {token_count} | {tps:.1f} tok/s | Total: {end - start:.2f}s", "INFO")

    llm_stream.last_spoken_text = "".join(spoken_chunks)

# ============================================================
# LLM WORKER THREAD
# ============================================================
class LLMWorker(QThread):
    """Thread-safe LLM processing"""
    text_chunk = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, user_text: str):
        super().__init__()
        self.user_text = user_text
        self._running = True
        self._mutex = QMutex()
        log(f"LLMWorker created: '{user_text}'")

    def stop(self):
        """Thread-safe stop"""
        with QMutexLocker(self._mutex):
            log("LLMWorker stop requested")
            self._running = False

    def is_running_safe(self) -> bool:
        """Thread-safe running check"""
        with QMutexLocker(self._mutex):
            return self._running

    def run(self):
        log("LLMWorker started")
        buffer = ""
        chunk_count = 0
        
        try:
            for token in llm_stream(self.user_text):
                if not self.is_running_safe():
                    log("LLMWorker stopped by flag")
                    break

                buffer += token
                
                # Emit on sentence boundaries
                if any(p in buffer for p in ".!?"):
                    chunk_count += 1
                    text = buffer.strip()
                    log(f"Emitting chunk #{chunk_count}: '{text}'")
                    self.text_chunk.emit(text)
                    buffer = ""

            # Emit remaining buffer
            if buffer.strip() and self.is_running_safe():
                chunk_count += 1
                log(f"Emitting final chunk #{chunk_count}: '{buffer.strip()}'")
                self.text_chunk.emit(buffer.strip())

            log(f"LLMWorker completed ({chunk_count} chunks)")
            
        except Exception as e:
            error_msg = f"LLMWorker error: {e}"
            log(error_msg, "ERROR")
            traceback.print_exc()
            self.error.emit(error_msg)
        finally:
            self.finished.emit()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_fixed_model_path(original_path: Path) -> Path:
    """Create fixed Live2D model file"""
    try:
        log(f"Processing model: {original_path}")
        fixed_path = original_path.with_stem(original_path.stem + "_fixed")
        
        if fixed_path.exists():
            log(f"Using cached fixed model: {fixed_path}")
            return fixed_path

        log("Creating fixed model file...")
        with open(original_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.pop("DefaultExpression", None)
        if "FileReferences" in data:
            data["FileReferences"].pop("DefaultExpression", None)

        with open(fixed_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        log(f"Fixed model saved: {fixed_path}")
        return fixed_path
        
    except Exception as e:
        log(f"Error fixing model path: {e}", "ERROR")
        traceback.print_exc()
        return original_path
    
# ============================================================
# AUDIO HELPER FUNCTIONS
# ============================================================

def pitch_resample(audio: np.ndarray, semitones: float = 1.0) -> np.ndarray:
    """
    Raise the pitch of audio by resampling.
    Positive semitones = higher pitch.

    Parameters:
        audio (np.ndarray): float32 audio array
        semitones (float): semitone shift
    Returns:
        np.ndarray: pitch-shifted audio
    """
    # Compute resampling factor
    factor = 2 ** (semitones / 12)

    # New length after resampling
    new_len = int(len(audio) / factor)

    # Interpolate to new length
    resampled = np.interp(
        np.linspace(0, len(audio), new_len),
        np.arange(len(audio)),
        audio
    ).astype(np.float32)

    return resampled

# ============================================================
# AUDIO WORKER
# ============================================================
class AudioWorker(QThread):
    """Thread-safe audio playback with RMS tracking"""
    volume_update = Signal(float)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.audio_queue = queue.Queue()
        self._mutex = QMutex()
        log("AudioWorker initialized")

    def add_audio(self, audio: np.ndarray):
        """Thread-safe audio queueing"""
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            self.audio_queue.put(audio)
            log(f"Audio queued: {len(audio)} samples (queue size: {self.audio_queue.qsize()})")
        except Exception as e:
            log(f"Error queueing audio: {e}", "ERROR")

    def stop(self):
        """Thread-safe stop"""
        with QMutexLocker(self._mutex):
            log("AudioWorker stop requested")
            self.running = False

    def is_running_safe(self) -> bool:
        """Thread-safe running check"""
        with QMutexLocker(self._mutex):
            return self.running

    def run(self):
        log("AudioWorker started")
        frame_count = 0
        
        try:
            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SIZE
            ) as stream:
                log(f"Audio stream opened (SR={SAMPLE_RATE}, BS={FRAME_SIZE})")

                while self.is_running_safe():
                    try:
                        audio = self.audio_queue.get(timeout=0.05)
                        log(f"Processing audio: {len(audio)} samples")

                        # Process in blocks
                        for i in range(0, len(audio), FRAME_SIZE):
                            if not self.is_running_safe():
                                log("AudioWorker stopped during playback")
                                break

                            frame = audio[i:i + FRAME_SIZE]
                            if len(frame) == 0:
                                continue

                            # Remove DC offset
                            frame = frame - np.mean(frame)
                            
                            # Play audio
                            stream.write(frame.reshape(-1, 1))

                            # Calculate RMS
                            rms = float(np.sqrt(np.mean(frame ** 2)))
                            
                            frame_count += 1
                            if frame_count % 100 == 0:
                                log(f"Frame {frame_count}, RMS: {rms:.6f}")

                            self.volume_update.emit(rms)

                    except queue.Empty:
                        # No audio, emit silence
                        self.volume_update.emit(0.0)

        except Exception as e:
            error_msg = f"AudioWorker error: {e}"
            log(error_msg, "ERROR")
            traceback.print_exc()
            self.error.emit(error_msg)

        log("AudioWorker exited")

    def stop_and_wait(self):
        """Safely stop and wait for thread"""
        self.stop()
        if not self.wait(5000):  # 5 second timeout
            log("AudioWorker did not stop in time", "WARNING")

# ============================================================
# TTS GENERATOR
# ============================================================
class TTSGenerator(QThread):
    """Thread-safe TTS generation"""
    audio_chunk = Signal(np.ndarray)
    finished_tts = Signal()
    error = Signal(str)

    def __init__(self, pipeline: KPipeline, text: str):
        super().__init__()
        self.pipeline = pipeline
        self.text = text
        self._mutex = QMutex()
        self._running = True
        log(f"TTSGenerator created: '{text}'")

    def stop(self):
        """Thread-safe stop"""
        with QMutexLocker(self._mutex):
            self._running = False

    def is_running_safe(self) -> bool:
        """Thread-safe running check"""
        with QMutexLocker(self._mutex):
            return self._running

    def run(self):
        log(f"TTSGenerator started: '{self.text}'")
        chunk_count = 0
        
        try:
            for r in self.pipeline(self.text, voice=TTS_VOICE):
                if not self.is_running_safe():
                    log("TTSGenerator stopped early")
                    break
                
                chunk_count += 1
                audio = (
                    r.output.audio
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                audio = pitch_resample(audio, semitones=5.0)  

                # Normalize
                audio = audio - np.mean(audio)
                np.clip(audio, -1.0, 1.0, out=audio)

                log(f"TTS chunk #{chunk_count}: {len(audio)} samples, range=[{audio.min():.3f}, {audio.max():.3f}]")
                self.audio_chunk.emit(audio)

            log(f"TTSGenerator completed ({chunk_count} chunks)")
            
        except Exception as e:
            error_msg = f"TTS error: {e}"
            log(error_msg, "ERROR")
            traceback.print_exc()
            self.error.emit(error_msg)
        finally:
            self.finished_tts.emit()

# ============================================================
# INPUT THREAD
# ============================================================
class InputThread(QThread):
    """Thread-safe console input"""
    text_received = Signal(str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self._mutex = QMutex()
        log("InputThread initialized")

    def stop(self):
        """Thread-safe stop"""
        with QMutexLocker(self._mutex):
            log("InputThread stop requested")
            self.running = False

    def is_running_safe(self) -> bool:
        """Thread-safe running check"""
        with QMutexLocker(self._mutex):
            return self.running

    def run(self):
        print("\n" + "="*60)
        print("KOKORO AI COMPANION")
        print("="*60)
        print("Type text and press Enter to talk")
        print("Commands: /quit or /exit to quit")
        print("="*60 + "\n")
        
        log("InputThread started, waiting for input")
        
        while self.is_running_safe():
            try:
                sys.stdout.flush()
                text = input("> ")
                
                if not text:
                    continue

                log(f"Input received: '{text}'")

                # Handle commands
                if text.strip().lower() in ("/quit", "/exit"):
                    log("Quit command received")
                    QCoreApplication.quit()
                    break

                self.text_received.emit(text)
                
            except EOFError:
                log("EOF received", "INFO")
                break
            except KeyboardInterrupt:
                log("KeyboardInterrupt received", "INFO")
                break
            except Exception as e:
                error_msg = f"InputThread error: {e}"
                log(error_msg, "ERROR")
                traceback.print_exc()
                self.error.emit(error_msg)
                break

        log("InputThread exited")

# ============================================================
# LIVE2D WIDGET
# ============================================================
class Live2DWidget(QOpenGLWidget):
    """Main widget with Live2D rendering and audio/TTS coordination"""
    
    def __init__(self):
        super().__init__()
        log("Live2DWidget initializing")

        # State
        self.model = None
        self.start_time = time.time()
        self.pipeline = None
        
        # Audio/mouth state
        self.mouth_raw = 0.0
        self.mouth_value = 0.0
        self._peak = 1e-6
        
        # TTS queue
        self.tts_queue = queue.Queue()
        self.tts_busy = False
        self._tts_start_time = None
        self._tts_mutex = QMutex()
        
        # First frame ghost fix
        self._first_frame_rendered = False
        self._ghost_fix_timer = None
        
        # Workers
        self.audio_worker = AudioWorker()
        self.audio_worker.volume_update.connect(self.on_volume)
        self.audio_worker.error.connect(self.on_error)
        
        # OpenGL format
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.NoProfile)
        fmt.setAlphaBufferSize(8)
        self.setFormat(fmt)
        
        # Window setup
        flags = (
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Window |
            Qt.BypassWindowManagerHint
        )
        self.setWindowFlags(flags)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Transparency attributes
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)
        self.setUpdateBehavior(QOpenGLWidget.NoPartialUpdate)
        
        self.setMouseTracking(True)
        self.drag_position = None

        # Update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // FPS)
        
        # macOS ghosting fix
        if sys.platform == "darwin":
            QGuiApplication.instance().applicationStateChanged.connect(self._on_app_state_changed)
        
        log(f"Live2DWidget initialized ({FPS} FPS)")

    def on_error(self, error_msg: str):
        """Handle errors from worker threads"""
        log(f"Widget received error: {error_msg}", "ERROR")

    def set_pipeline(self, pipeline: KPipeline):
        """Set TTS pipeline"""
        log("TTS pipeline set")
        self.pipeline = pipeline

    def enqueue_tts(self, text: str):
        """Handle new user input"""
        log(f"\n{'='*60}")
        log(f"New input: '{text}'")
        log('='*60)
        
        try:
            # Stop existing LLM worker
            if hasattr(self, "llm_worker") and self.llm_worker.isRunning():
                log("Stopping previous LLM worker")
                self.llm_worker.stop()
                if not self.llm_worker.wait(3000):
                    log("LLM worker did not stop in time", "WARNING")

            # Stop existing TTS
            if hasattr(self, "current_tts") and self.current_tts.isRunning():
                log("Stopping current TTS")
                self.current_tts.stop()
                if not self.current_tts.wait(3000):
                    log("TTS did not stop in time", "WARNING")
            
            # Clear TTS queue
            cleared = 0
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
            
            if cleared > 0:
                log(f"Cleared {cleared} items from TTS queue")
            
            with QMutexLocker(self._tts_mutex):
                self.tts_busy = False

            # Start new LLM worker
            self.llm_worker = LLMWorker(text)
            self.llm_worker.text_chunk.connect(self.on_llm_chunk)
            self.llm_worker.finished.connect(self.on_llm_finished)
            self.llm_worker.error.connect(self.on_error)
            self.llm_worker.start()
            
            log("LLM worker started")
            
        except Exception as e:
            log(f"Error in enqueue_tts: {e}", "ERROR")
            traceback.print_exc()

    def on_llm_chunk(self, text: str):
        """Handle LLM chunk - add to TTS queue"""
        try:
            self.tts_queue.put(text)
            log(f"Added to TTS queue (size: {self.tts_queue.qsize()})")
            self.process_queue()
        except Exception as e:
            log(f"Error handling LLM chunk: {e}", "ERROR")

    def on_llm_finished(self):
        """Handle LLM completion"""
        log("LLM generation finished")
        self.process_queue()

    def process_queue(self):
        """Process next item in TTS queue"""
        with QMutexLocker(self._tts_mutex):
            is_busy = self.tts_busy
            queue_size = self.tts_queue.qsize()
        
        log(f"process_queue: busy={is_busy}, queue_size={queue_size}")
        
        if is_busy or queue_size == 0:
            return  
        
        try:
            # Stop any running TTS first
            if hasattr(self, "current_tts") and self.current_tts.isRunning():
                log("Stopping previous TTS")
                self.current_tts.stop()
                if not self.current_tts.wait(2000):
                    log("Previous TTS did not stop in time", "WARNING")

            with QMutexLocker(self._tts_mutex):
                self.tts_busy = True
            
            text = self.tts_queue.get()
            log(f"Starting TTS: '{text}'")

            if not self.pipeline:
                log("No TTS pipeline available", "ERROR")
                with QMutexLocker(self._tts_mutex):
                    self.tts_busy = False
                return

            self.current_tts = TTSGenerator(self.pipeline, text)
            self.current_tts.audio_chunk.connect(self.audio_worker.add_audio)
            self.current_tts.finished_tts.connect(self.on_tts_done)
            self.current_tts.error.connect(self.on_error)

            self._tts_start_time = time.time()
            self.current_tts.start()
            
        except Exception as e:
            log(f"Error in process_queue: {e}", "ERROR")
            traceback.print_exc()
            with QMutexLocker(self._tts_mutex):
                self.tts_busy = False

    def on_tts_done(self):
        """Handle TTS completion"""
        try:
            if self._tts_start_time:
                elapsed = time.time() - self._tts_start_time
                log(f"TTS completed in {elapsed:.2f}s", "INFO")

            self._tts_start_time = None
            
            with QMutexLocker(self._tts_mutex):
                self.tts_busy = False
            
            self.process_queue()
            
        except Exception as e:
            log(f"Error in on_tts_done: {e}", "ERROR")
            traceback.print_exc()

    def initializeGL(self):
        """Initialize OpenGL and Live2D"""
        try:
            log("Initializing OpenGL")
            live2d.glInit()
            glClearColor(0, 0, 0, 0)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            path = get_fixed_model_path(Path(LIVE2D_MODEL_PATH))
            log(f"Loading Live2D model: {path}")

            self.model = live2d.LAppModel()
            self.model.LoadModelJson(str(path))
            self.model.Resize(self.width(), self.height())
            
            log(f"Live2D model loaded ({self.width()}x{self.height()})")

            self.audio_worker.start()
            log("Audio worker started")
            
        except Exception as e:
            log(f"Error in initializeGL: {e}", "ERROR")
            traceback.print_exc()

    def resizeGL(self, w: int, h: int):
        """Handle window resizing"""
        if self.model:
            self.model.Resize(w, h)

    def on_volume(self, v_raw: float):
        """Process audio volume for lip sync"""
        try:
            # Floor noise
            v = max(0.0, v_raw - AUDIO_NOISE_FLOOR)

            # Compress and gain
            v = v * AUDIO_VOLUME_GAIN
            v = v ** AUDIO_COMPRESSION

            # Track peak
            self._peak = max(self._peak * 0.995, max(v, 1e-6))

            # Normalize
            v = v / (self._peak + 1e-6)
            v = min(max(v * LIP_SYNC_SENSITIVITY, 0.0), 1.0)

            # Smooth
            self.mouth_raw = max(self.mouth_raw * 0.85, v)
            
        except Exception as e:
            log(f"Error in on_volume: {e}", "ERROR")

    def _apply_first_frame_fix(self):
        """Restore model to normal size after initial ghost frame"""
        try:
            if self.model:
                # Reset to normal scale
                self.model.SetParameterValue("ParamBodyScaleX", 1.0)
                self.model.SetParameterValue("ParamBodyScaleY", 1.0)
                self._first_frame_rendered = True
                log("First frame ghost fix applied - model restored to normal size", "INFO")
        except Exception as e:
            log(f"Error in first frame fix: {e}", "ERROR")

    def paintGL(self):
        """Render Live2D model"""
        try:
            live2d.clearBuffer()
            if not self.model:
                return

            # First frame ghost fix for macOS Metal
            if not self._first_frame_rendered and sys.platform == "darwin":
                log("Rendering first frame at tiny scale to prevent ghost", "INFO")
                # Scale model to nearly invisible for first frame
                self.model.SetParameterValue("ParamBodyScaleX", 0.001)
                self.model.SetParameterValue("ParamBodyScaleY", 0.001)
                
                # Schedule restoration to normal size
                if not self._ghost_fix_timer:
                    self._ghost_fix_timer = QTimer(self)
                    self._ghost_fix_timer.setSingleShot(True)
                    self._ghost_fix_timer.timeout.connect(self._apply_first_frame_fix)
                    self._ghost_fix_timer.start(100)  # 100ms delay

            # Breathing animation
            t = time.time() - self.start_time
            self.model.SetParameterValue("PARAM_BREATH", (math.sin(t * 2) + 1) / 2)

            # Mouth animation
            target = min(self.mouth_raw, 1.0) if self.mouth_raw > LIP_SYNC_THRESHOLD else 0.0
            self.mouth_value += (target - self.mouth_value) * LIP_SYNC_SMOOTHING

            for p in ("PARAM_MOUTH_OPEN_Y", "ParamMouthOpenY"):
                self.model.SetParameterValue(p, self.mouth_value)

            self.model.Update()
            self.model.Draw()
            
        except Exception as e:
            log(f"Error in paintGL: {e}", "ERROR")

    def _on_app_state_changed(self, state):
        """Forces a redraw on app activation to fix ghosting on macOS"""
        if state != Qt.ApplicationActive or not self.isValid():
            return

        try:
            self.makeCurrent()
            glFinish()

            w, h = self.width(), self.height()
            super().resizeGL(w + 1, h + 1)
            super().resizeGL(w, h)

            for _ in range(2):
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                live2d.clearBuffer()
                self.update()
                
            log("macOS ghost fix applied (app state changed)", "INFO")
        except Exception as e:
            log(f"Error in _on_app_state_changed: {e}", "ERROR")

    def showEvent(self, event):
        """Ensure first frame draws correctly"""
        super().showEvent(event)
        QTimer.singleShot(0, self.update)

    def closeEvent(self, event):
        """Clean shutdown"""
        log("Shutting down Live2DWidget")
        
        try:
            # Stop ghost fix timer
            if self._ghost_fix_timer:
                self._ghost_fix_timer.stop()
            
            # Stop LLM worker
            if hasattr(self, "llm_worker") and self.llm_worker.isRunning():
                log("Stopping LLM worker")
                self.llm_worker.stop()
                self.llm_worker.wait(3000)

            # Stop TTS
            if hasattr(self, "current_tts") and self.current_tts.isRunning():
                log("Stopping TTS")
                self.current_tts.stop()
                self.current_tts.wait(3000)

            # Stop audio worker
            log("Stopping audio worker")
            self.audio_worker.stop_and_wait()
            
            event.accept()
            log("Shutdown complete")
            
        except Exception as e:
            log(f"Error during shutdown: {e}", "ERROR")
            traceback.print_exc()
            event.accept()

    # Mouse event handlers for dragging
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_position is not None:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = None
            event.accept()

# ============================================================
# MAIN
# ============================================================
def main():
    """Main entry point with comprehensive error handling"""
    log("="*60, "INFO")
    log("KOKORO AI COMPANION STARTING", "INFO")
    log("="*60, "INFO")
    
    # Signal handlers
    def signal_handler(signum, frame):
        log(f"Signal {signum} received, shutting down...", "INFO")
        QCoreApplication.quit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Environment
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    exit_code = 0
    
    try:
        log("Creating QApplication")
        app = QApplication(sys.argv)
        
        # Set global OpenGL format for transparency
        fmt = QSurfaceFormat()
        fmt.setAlphaBufferSize(8)
        QSurfaceFormat.setDefaultFormat(fmt)

        # Initialize TTS pipeline
        log("Initializing TTS pipeline...")
        pipeline = KPipeline(lang_code="a", device="cpu")
        log("TTS pipeline ready")

        # Initialize Live2D
        log("Initializing Live2D framework...")
        live2d.init()
        log("Live2D ready")

        # Create widget
        log("Creating Live2D widget...")
        widget = Live2DWidget()
        widget.set_pipeline(pipeline)
        widget.show()
        log("Widget displayed")

        # Start input thread
        log("Starting input thread...")
        input_thread = InputThread()
        input_thread.text_received.connect(widget.enqueue_tts)
        input_thread.error.connect(widget.on_error)
        input_thread.start()
        log("Input thread started")

        log("="*60, "INFO")
        log("READY - System operational", "INFO")
        log("="*60, "INFO")

        # Run event loop
        exit_code = app.exec()
        log(f"Event loop exited with code: {exit_code}", "INFO")
        
    except Exception as e:
        log(f"Fatal error in main: {e}", "ERROR")
        traceback.print_exc()
        exit_code = 1
        
    finally:
        log("Cleanup starting...", "INFO")
        
        try:
            if 'input_thread' in locals():
                input_thread.stop()
                if not input_thread.wait(3000):
                    log("Input thread did not stop in time", "WARNING")
        except:
            pass
        
        try:
            live2d.dispose()
        except:
            pass
        
        log("="*60, "INFO")
        log("KOKORO AI COMPANION STOPPED", "INFO")
        log("="*60, "INFO")
        
    sys.exit(exit_code)

if __name__ == "__main__":
    main()