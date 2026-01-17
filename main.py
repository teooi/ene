# main.py
import sys
import os
from PySide6.QtWidgets import QApplication
import live2d.v3 as live2d

from llm.model import init_llm, load_system_prompt
from live2d_ui import Live2DWidget
from input.input_thread import InputThread
from kokoro import KPipeline

def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    app = QApplication(sys.argv)

    live2d.init()

    llm = init_llm()
    system_prompt = load_system_prompt()

    pipeline = KPipeline(lang_code="a", device="mps")

    widget = Live2DWidget(llm, system_prompt)
    widget.set_pipeline(pipeline)
    widget.show()

    input_thread = InputThread()
    input_thread.text_received.connect(widget.enqueue_tts)
    input_thread.start()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()

