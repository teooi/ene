# tts/tts_worker.py

from __future__ import annotations

import traceback

import numpy as np

from PySide6.QtCore import (
    QThread,
    Signal,
    QMutex,
    QMutexLocker,
)

from config import (
    TTS_VOICE
)

from tts.audio_utils import pitch_resample

from logger import log

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

                audio = pitch_resample(audio, semitones=4.0)  

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

