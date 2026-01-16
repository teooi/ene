# input/input_thread.py

from __future__ import annotations

import sys
import traceback

from PySide6.QtCore import (
    QThread,
    Signal,
    QMutex,
    QMutexLocker,
    Qt,
    QCoreApplication,
)

from logger import log

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
