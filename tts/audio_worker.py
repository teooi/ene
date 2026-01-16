import queue
import traceback
import numpy as np
import sounddevice as sd

from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from config import SAMPLE_RATE, FRAME_SIZE
from logger import log

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

