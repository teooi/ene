import sys
import os
import math
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QGuiApplication, QSurfaceFormat

from OpenGL.GL import *
import live2d.v3 as live2d

# ---------------- Configuration ----------------
# Reduced from 800 to 400. Smaller windows are easier to manage as "pets".
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
FPS = 60

# ---------------- Live2D Widget ----------------
class Live2DWidget(QOpenGLWidget):
    """A transparent, frameless widget to render Live2D models."""

    def __init__(self, model_path: Path, parent=None):
        super().__init__(parent)
        
        self.model = None
        self.model_path = model_path
        self.start_time = time.time()
        self.drag_position = None

        # 1. Setup OpenGL Format
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setVersion(2, 1) 
        fmt.setProfile(QSurfaceFormat.NoProfile)
        fmt.setAlphaBufferSize(8)
        self.setFormat(fmt)

        # 2. Window Configuration
        flags = (
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Window |
            Qt.BypassWindowManagerHint  
        )
        self.setWindowFlags(flags)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # 3. Transparency Attributes
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)
        self.setUpdateBehavior(QOpenGLWidget.NoPartialUpdate)

        # 4. Animation Loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // FPS)

        # 5. macOS Ghosting Fix Listener
        if sys.platform == "darwin":
            QGuiApplication.instance().applicationStateChanged.connect(self._on_app_state_changed)

    # ---------------- Mouse Events for Dragging ----------------
    def mousePressEvent(self, event):
        """Record the starting position of the mouse drag."""
        if event.button() == Qt.LeftButton:
            # Calculate offset from the top-left of the window
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Move the window if the left mouse button is held down."""
        if event.buttons() == Qt.LeftButton and self.drag_position is not None:
            # Move window to new global position minus the offset
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        """Reset drag position on release."""
        if event.button() == Qt.LeftButton:
            self.drag_position = None
            event.accept()

    # ---------------- OpenGL & Live2D ----------------
    def initializeGL(self):
        """Initialize OpenGL resources and load the model."""
        live2d.glInit()
        glClearColor(0, 0, 0, 0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = live2d.LAppModel()
        self.model.LoadModelJson(str(self.model_path))
        self.model.Resize(self.width(), self.height())
        
        self.model.SetParameterValue("PARAM_ANGLE_X", 0.0)
        self.model.SetParameterValue("PARAM_ANGLE_Y", 0.0)

    def paintGL(self):
        """Main rendering loop."""
        live2d.clearBuffer()
        if not self.model:
            return

        t = time.time() - self.start_time

        self.model.SetParameterValue("PARAM_ANGLE_X", math.sin(t) * 30)
        self.model.SetParameterValue("PARAM_ANGLE_Y", math.sin(t / 2) * 10)

        blink = (math.sin(t * 3) + 1) / 2
        self.model.SetParameterValue("PARAM_EYE_L_OPEN", blink)
        self.model.SetParameterValue("PARAM_EYE_R_OPEN", blink)

        self.model.Update()
        self.model.Draw()

    def resizeGL(self, w: int, h: int):
        """Handle window resizing."""
        if self.model:
            self.model.Resize(w, h)

    # ---------------- macOS Workarounds ----------------
    def _on_app_state_changed(self, state):
        """Forces a redraw on app activation to fix 'ghosting' on macOS."""
        if state != Qt.ApplicationActive or not self.isValid():
            return

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
        """Ensure first frame draws correctly."""
        super().showEvent(event)
        QTimer.singleShot(0, self.update)


# ---------------- Main Entry Point ----------------
def main():
    # 1. High DPI Fix
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    
    # Set global OpenGL format
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)

    # 3. Initialize Live2D
    try:
        live2d.init()
    except Exception as e:
        print(f"Failed to initialize Live2D: {e}")
        sys.exit(1)

    # 4. Setup Model Path
    model_json_path = Path("pichu/Pichu.model3.json")

    if not model_json_path.exists():
        print(f"Error: Model not found at {model_json_path}")
        sys.exit(1)

    widget = Live2DWidget(model_json_path)
    widget.show()

    try:
        sys.exit(app.exec())
    finally:
        live2d.dispose()

if __name__ == "__main__":
    main()