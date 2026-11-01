"""SignBangla Module 2: Live Speech-to-Sign Language Synthesis.

Listen to spoken Bengali (or English) via the microphone, transcribe with ASR,
query the 60-word BdSLW60 sign dictionary, stitch keypoint sequences with smooth
easing transitions, and render the animated skeleton video with native Bangla captions.

Usage:
    # 1. Speak into microphone (default 4-second audio window, Bengali ASR):
    python scripts/21_speech_to_sign_live.py

    # 2. Speak in English (maps English words like 'father apple' to sign dictionary):
    python scripts/21_speech_to_sign_live.py --lang en-US

    # 3. Increase/decrease listening duration:
    python scripts/21_speech_to_sign_live.py --duration 5.0

    # 4. Direct text input without microphone:
    python scripts/21_speech_to_sign_live.py --text "বাবা আপেল"
"""
import argparse
import io
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import _path  # noqa: F401
import cv2
import numpy as np

try:
    import sounddevice as sd
    import speech_recognition as sr
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

from signbangla.speech2sign.dictionary import SignDictionary
from signbangla.speech2sign.render import render_frames, render_video
from signbangla.speech2sign.stitch import EASINGS, frame_captions, stitch_words
from signbangla.utils import ROOT, load_config, work_dir


def record_speech_audio(duration: float = 4.0, samplerate: int = 16000, lang: str = "bn-BD") -> str:
    """Record audio from the default microphone and transcribe using SpeechRecognition."""
    if not HAS_AUDIO:
        print("[!] Audio dependencies not available. Please enter text manually.")
        return ""

    print(f"\n============================================================")
    print(f"      SignBangla Module 2: Live Speech Input                ")
    print(f"============================================================")
    print(f"[*] Language: {lang} | Recording duration: {duration:.1f}s")
    print(f"[*] Initializing microphone...")

    # Record audio via sounddevice
    print(f"\n>>> [LISTENING NOW - PLEASE SPEAK CLEARLY] <<<")
    for sec in range(int(duration), 0, -1):
        print(f"    Recording... {sec}s remaining", end="\r", flush=True)
        time.sleep(0.01)

    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    print("    Recording complete! Processing audio...                  \n")

    # Encode to WAV buffer in memory
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())
    wav_io.seek(0)

    # Transcribe with Google Speech Recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language=lang)
        print(f"[✓] Transcribed Speech: \"{text}\"")
        return text.strip()
    except sr.UnknownValueError:
        print("[!] Speech recognition could not understand audio (too quiet or unclear).")
        return ""
    except sr.RequestError as e:
        print(f"[!] Speech recognition service error: {e}")
        return ""


def main():
    ap = argparse.ArgumentParser(description="Live Speech-to-Sign Language Synthesis")
    ap.add_argument("--config", help="Path to config YAML")
    ap.add_argument("--text", help="Direct text input (skips microphone)")
    ap.add_argument("--lang", default="bn-BD", help="ASR language code: 'bn-BD' (Bangla) or 'en-US' (English)")
    ap.add_argument("--duration", type=float, default=4.0, help="Microphone recording duration in seconds")
    ap.add_argument("--n-transition", type=int, default=12, help="Transition frames between signs")
    ap.add_argument("--easing", default="cosine", choices=sorted(EASINGS), help="Transition easing function")
    ap.add_argument("--variant", type=int, default=0, help="Trial variant to choose (0, 1, or 2)")
    ap.add_argument("--out", default="work/m2_speech_output.mp4", help="Output MP4 file path")
    ap.add_argument("--no-open", action="store_true", help="Do not automatically open video on macOS")
    args = ap.parse_args()

    cfg = load_config(args.config)
    s2s = cfg["speech2sign"]
    d = SignDictionary.from_config(cfg)

    # 1. Acquire Speech or Text
    input_text = args.text
    if not input_text:
        input_text = record_speech_audio(duration=args.duration, lang=args.lang)

    # Fallback if speech transcription was empty
    if not input_text:
        print("\nAvailable sample vocabulary words: বাবা (Father), আপেল (Apple), মা (Mother), চা (Tea), ইত্যাদি")
        input_text = input("Enter Bangla or English words to sign: ").strip()

    if not input_text:
        print("No input provided. Exiting.")
        sys.exit(0)

    print(f"\n[*] Processing sentence: \"{input_text}\"")

    # 2. Tokenize sentence against sign dictionary
    try:
        words = d.tokenize(input_text)
    except KeyError as e:
        print(f"[!] Tokenization error: {e}")
        print("    Try using standard vocabulary words from the 60-word BdSLW60 dictionary.")
        sys.exit(1)

    print(f"[✓] Recognized {len(words)} sign gesture(s): {words}")

    # 3. Stitch sequence with smooth transitions
    params = dict(s2s["stitch"])
    params["n_transition"] = args.n_transition
    params["easing"] = args.easing

    st = stitch_words(words, d, variant=args.variant, **params)
    captions = frame_captions(st, input_text)
    font_path = ROOT / s2s["font"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. Render H.264 Video
    print(f"[*] Rendering H.264 skeleton animation with native Bangla captions...")
    n_frames = render_video(
        st.seq,
        out_path,
        fps=d.fps,
        captions=captions,
        font_path=font_path,
        **s2s["render"]
    )

    print(f"[✓] Rendered {n_frames} frames ({n_frames / d.fps:.1f}s at {d.fps} FPS)")
    print(f"[✓] Video saved to: {out_path.resolve()}\n")

    # Print segment breakdown
    print("Timeline Segments:")
    for seg in st.segments:
        name = d.words[d.find(seg["label"])]["romanized"][0] if seg["label"] else ""
        print(f"  Frames {seg['start']:4d} - {seg['end']:4d} | {seg['kind']:10s} | {name}")

    # 5. Automatically play video in macOS default player (QuickTime)
    if sys.platform == "darwin" and not args.no_open and out_path.exists():
        print(f"\n[*] Opening {out_path.name} in macOS QuickTime Player...")
        subprocess.run(["open", str(out_path)])

    print("\n============================================================")
    print("      Module 2 Speech-to-Sign Completed Successfully!       ")
    print("============================================================\n")


if __name__ == "__main__":
    main()
