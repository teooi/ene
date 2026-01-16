import sys
import time
import math
import queue
import traceback
from pathlib import Path

from kokoro import KPipeline

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QMutex, QMutexLocker
from PySide6.QtGui import QSurfaceFormat, QGuiApplication

from OpenGL.GL import *

import live2d.v3 as live2d  

from logger import log
from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    FPS,
    LIP_SYNC_SENSITIVITY,
    LIP_SYNC_SMOOTHING,
    LIP_SYNC_THRESHOLD,
    AUDIO_NOISE_FLOOR,
    AUDIO_VOLUME_GAIN,
    AUDIO_COMPRESSION,
    SAMPLE_RATE,
    FRAME_SIZE,
    LIVE2D_MODEL_PATH,
)

from tts import AudioWorker, TTSGenerator
from llm import LLMWorker
from .model_utils import get_fixed_model_path

class Live2DWidget(QOpenGLWidget):
    """Main widget with Live2D rendering and audio/TTS coordination"""
    
    def __init__(self, llm, system_prompt, parent=None):
        super().__init__(parent)

        self.llm = llm
        self.system_prompt = system_prompt
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
            self.llm_worker = LLMWorker(
                self.llm,
                self.system_prompt,
                text,
            )
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
