import numpy as np

def pitch_resample(audio: np.ndarray, semitones: float = 1.0) -> np.ndarray:
    factor = 2 ** (semitones / 12)
    new_len = int(len(audio) / factor)
    return np.interp(
        np.linspace(0, len(audio), new_len),
        np.arange(len(audio)),
        audio
    ).astype(np.float32)

