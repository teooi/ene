import sys
import os
import math
import time
import json
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QPoint, QThread, Signal
from PySide6.QtGui import QGuiApplication, QSurfaceFormat, QCursor

from OpenGL.GL import *
import live2d.v3 as live2d

# Audio handling
import sounddevice as sd
import numpy as np

# ---------------- Configuration ----------------
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
FPS = 60
AUDIO_DEVICE_NAME = None 

# Debug Settings
LIP_SYNC_SENSITIVITY = 2.0
LIP_SYNC_SMOOTHING = 0.3
# Lowered threshold to pick up everything for debugging
LIP_SYNC_THRESHOLD = 0.01 

# ---------------- Helper: Fix Model JSON ----------------
def get_fixed_model_path(original_path):
    try:
        with open(original_path, 'r') as f:
            data = json.load(f)
        changed = False
        if 'DefaultExpression' in data and data['DefaultExpression']:
            data['DefaultExpression'] = "" 
            changed = True
        elif 'FileReferences' in data and 'DefaultExpression' in data['FileReferences']:
             data['FileReferences']['DefaultExpression'] = ""
             changed = True
        if changed:
            fixed_path = original_path.parent / f"{original_path.stem}_fixed{original_path.suffix}"
            with open(fixed_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"[Model Fix] Created fixed config: {fixed_path}")
            return fixed_path
        else:
            return original_path
    except Exception as e:
        print(f"[Model Fix] Error reading JSON: {e}")
        return original_path

# ---------------- Audio Worker Thread ----------------
class AudioWorker(QThread):
    volume_update = Signal(float)

    def __init__(self, device_name=None):
        super().__init__()
        self.running = True
        self.device_name = device_name
        self.stream = None

    def run(self):
        device_id = None
        if self.device_name:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if self.device_name.lower() in d['name'].lower():
                    if d['max_input_channels'] > 0:
                        device_id = i
                        break
        if device_id is None:
            device_id = sd.default.device[0]

        print(f"[Audio] Listening to: {sd.query_devices(device_id)['name']}")

        try:
            self.stream = sd.InputStream(
                device=device_id, channels=1, samplerate=44100, dtype='float32', blocksize=1024
            )
            self.stream.start()
            frame_count = 0
            while self.running:
                data, overflowed = self.stream.read(1024)
                rms = np.sqrt(np.mean(data**2))
                volume = min(rms * 50.0, 1.0) 
                
                # --- DEBUG PRINT: Log Volume ---
                # Print volume every 30 frames (approx 2-3 times per second) to avoid spamming
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[Audio] Raw Volume: {volume:.4f} (Threshold: {LIP_SYNC_THRESHOLD})")

                self.volume_update.emit(volume)
        except Exception as e:
            print(f"[Audio] Error: {e}")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()

    def stop(self):
        self.running = False
        self.wait()

# ---------------- Live2D Widget ----------------
class Live2DWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.model = None
        self.start_time = time.time()
        self.drag_position = None
        
        # Mouth State
        self.mouth_raw_volume = 0.0
        self.mouth_open_value = 0.0 
        
        # Manual Test Mode (Force mouth open)
        self.manual_mouth_open = False
        
        self.expressions = []
        self.setMouseTracking(True)

        self.audio_thread = AudioWorker(AUDIO_DEVICE_NAME)
        self.audio_thread.volume_update.connect(self.on_audio_volume_received)

        # OpenGL Setup
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.NoProfile)
        fmt.setAlphaBufferSize(8)
        self.setFormat(fmt)

        # Window Setup
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Window | Qt.BypassWindowManagerHint
        self.setWindowFlags(flags)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)
        self.setUpdateBehavior(QOpenGLWidget.NoPartialUpdate)

        # Loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // FPS)

        if sys.platform == "darwin":
            QGuiApplication.instance().applicationStateChanged.connect(self._on_app_state_changed)

    def initializeGL(self):
        live2d.glInit()
        glClearColor(0, 0, 0, 0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        original_model_path = Path("pichu/Pichu.model3.json")
        if not original_model_path.exists():
            print(f"Error: Model not found at {original_model_path}")
            return

        model_to_load = get_fixed_model_path(original_model_path)
        self.model = live2d.LAppModel()
        self.model.LoadModelJson(str(model_to_load))
        self.model.Resize(self.width(), self.height())
        
        exp_ids = self.model.GetExpressionIds()
        if exp_ids:
            if isinstance(exp_ids, dict):
                self.expressions = list(exp_ids.keys())
            else:
                self.expressions = list(exp_ids)
        
        print(f"[System] Loaded Pichu. Available Expressions:")
        for i, name in enumerate(self.expressions):
            print(f"  [{i+1}] {name}")
        print(f"  [0] Default/Neutral")
        print(f"[System] MANUAL TEST: Hold [SPACEBAR] to force mouth open.")
            
        self.audio_thread.start()

    def keyPressEvent(self, event):
        key = event.key()
        
        # TEST MODE: Force mouth open
        if key == Qt.Key_Space:
            self.manual_mouth_open = True
            print("[Test] Manual Mouth Open: ON")

        # Expressions
        if key == Qt.Key_0:
            self.model.SetExpression("") 
            print("[System] Reset to Default Expression")
        elif Qt.Key_1 <= key <= Qt.Key_9:
            index = key - Qt.Key_1
            if index < len(self.expressions):
                exp_name = self.expressions[index]
                self.model.SetExpression(exp_name)
                print(f"[System] Set expression: {exp_name}")

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.manual_mouth_open = False
            print("[Test] Manual Mouth Open: OFF")

    # --- Mouse Events ---
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_position is not None:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = None
            event.accept()

    def on_audio_volume_received(self, volume):
        self.mouth_raw_volume = volume

    def paintGL(self):
        live2d.clearBuffer()
        if not self.model: return

        t = time.time() - self.start_time

        # --- GLOBAL MOUSE TRACKING ---
        global_pos = QCursor.pos()
        local_pos = self.mapFromGlobal(global_pos)
        center_x, center_y = self.width() / 2, self.height() / 2
        mouse_x = (local_pos.x() - center_x) / center_x
        mouse_y = (local_pos.y() - center_y) / center_y

        # --- ANIMATION ---
        
        # 1. Breathing
        self.model.SetParameterValue("PARAM_BREATH", (math.sin(t * 2) + 1) / 2)

        # 2. Eye Tracking
        self.model.SetParameterValue("PARAM_ANGLE_X", mouse_x * 30.0)
        self.model.SetParameterValue("PARAM_ANGLE_Y", mouse_y * 30.0)
        self.model.SetParameterValue("PARAM_EYE_BALL_X", mouse_x)
        self.model.SetParameterValue("PARAM_EYE_BALL_Y", mouse_y)

        # 3. Mouth Sync Logic
        target = 0.0
        
        # PRIORITY 1: Manual Test (Spacebar)
        if self.manual_mouth_open:
            target = 1.0
        
        # PRIORITY 2: Audio
        elif self.mouth_raw_volume >= LIP_SYNC_THRESHOLD:
            target = (self.mouth_raw_volume * LIP_SYNC_SENSITIVITY)
            if target > 1.0: target = 1.0
        else:
            target = 0.0

        # Smoothing
        self.mouth_open_value += (target - self.mouth_open_value) * LIP_SYNC_SMOOTHING

        # Send to Model
        param_value = self.mouth_open_value
        
        # Log final value only if it's moving significantly
        if param_value > 0.01:
            print(f"[Mouth] Sent to Model: {param_value:.2f}")

        self.model.SetParameterValue("PARAM_MOUTH_OPEN_Y", param_value)
        self.model.SetParameterValue("PARAM_MOUTH_OPEN_X", param_value * 0.5)
        self.model.SetParameterValue("ParamMouthOpenY", param_value)
        self.model.SetParameterValue("Mouth Open", param_value)

        self.model.Update()
        self.model.Draw()

    def resizeGL(self, w, h):
        if self.model: self.model.Resize(w, h)

    def _on_app_state_changed(self, state):
        if state != Qt.ApplicationActive or not self.isValid(): return
        self.makeCurrent()
        glFinish()
        w, h = self.width(), self.height()
        super().resizeGL(w + 1, h + 1)
        super().resizeGL(w, h)
        for _ in range(2):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            live2d.clearBuffer()
            self.update()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.update)

    def closeEvent(self, event):
        self.audio_thread.stop()
        super().closeEvent(event)

# ---------------- Main ----------------
def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)

    try:
        live2d.init()
    except Exception as e:
        print(f"Failed to initialize Live2D: {e}")
        sys.exit(1)

    widget = Live2DWidget()
    widget.show()

    try:
        sys.exit(app.exec())
    finally:
        live2d.dispose()

if __name__ == "__main__":
    main()