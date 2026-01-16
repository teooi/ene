import traceback
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from logger import log
from .stream import llm_stream

class LLMWorker(QThread):
    """Thread-safe LLM processing"""

    text_chunk = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, llm, system_prompt, user_text, parent=None):
        super().__init__(parent)

        self.llm = llm
        self.system_prompt = system_prompt
        self.user_text = user_text

        self._running = True
        self._mutex = QMutex()

        log(f"LLMWorker created: '{user_text}'")

    def stop(self):
        with QMutexLocker(self._mutex):
            log("LLMWorker stop requested")
            self._running = False

    def is_running_safe(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    def run(self):
        log("LLMWorker started")
        buffer = ""
        chunk_count = 0

        try:
            for token in llm_stream(
                self.llm,
                self.system_prompt,
                self.user_text,
            ):
                if not self.is_running_safe():
                    log("LLMWorker stopped by flag")
                    break

                buffer += token

                if any(p in buffer for p in ".!?"):
                    chunk_count += 1
                    text = buffer.strip()
                    log(f"Emitting chunk #{chunk_count}: '{text}'")
                    self.text_chunk.emit(text)
                    buffer = ""

            if buffer.strip() and self.is_running_safe():
                chunk_count += 1
                self.text_chunk.emit(buffer.strip())

            log(f"LLMWorker completed ({chunk_count} chunks)")

        except Exception as e:
            msg = f"LLMWorker error: {e}"
            log(msg, "ERROR")
            traceback.print_exc()
            self.error.emit(msg)
        finally:
            self.finished.emit()
