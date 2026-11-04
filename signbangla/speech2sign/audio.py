"""Push-to-talk microphone recording: press Enter to start, Enter to stop. 16 kHz mono float32, what Whisper expects."""
import numpy as np
import sounddevice as sd


def record(sample_rate=16000, device=None, wait=input):
    """Record from the microphone. `wait` blocks until the user presses Enter. Returns float32 audio, shape (n,)."""
    chunks = []

    def on_audio(indata, frames, time, status):
        chunks.append(indata[:, 0].copy())

    wait("Press Enter to start recording... ")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", device=device, callback=on_audio):
        wait("Recording. Press Enter to stop... ")
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def loudest_rms(audio, sample_rate=16000, window_s=0.03):
    """Volume of the loudest short window, so one word in a long recording still counts as loud."""
    n = max(1, int(window_s * sample_rate))
    if len(audio) < n:
        return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    usable = audio[:len(audio) - len(audio) % n].reshape(-1, n)
    return float(np.sqrt(np.square(usable).mean(axis=1)).max())


def has_speech(audio, sample_rate=16000, min_seconds=0.3, min_rms=0.01):
    """False for recordings too short or too quiet to hold a word. Whisper invents text for these, so they must not
    reach it."""
    return len(audio) >= min_seconds * sample_rate and loudest_rms(audio, sample_rate) >= min_rms
