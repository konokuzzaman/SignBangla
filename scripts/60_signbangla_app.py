"""SignBangla app: one local page with both directions of communication.

    python scripts/60_signbangla_app.py                 (loads everything once, then opens the browser)
    python scripts/60_signbangla_app.py --no-browser
    python scripts/60_signbangla_app.py --no-asr        (typed text only: no speech model, faster start)
    python scripts/60_signbangla_app.py --camera 1      (another camera; --no-camera: word buttons only)

Settings: configs/config.yaml -> app. A missing camera, microphone, sign model, speech model or internet connection
only switches that part off; the page still opens and says what is missing. Ctrl+C stops the server.
"""
import argparse
import sys
import time

import _path  # noqa: F401

from signbangla.app.pipeline_sign2text import SignToText
from signbangla.app.pipeline_speech2sign import SpeechToSign
from signbangla.app.server import AppServer
from signbangla.utils import ROOT, load_config


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-asr", action="store_true", help="do not load the speech model (typed text still works)")
    ap.add_argument("--camera", type=int, help="camera index (default: app.sign.camera_index)")
    ap.add_argument("--no-camera", action="store_true", help="no camera (the word buttons still work)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    started = time.perf_counter()
    print("Loading Speech -> Sign (dictionary, speech model) ...", flush=True)
    speech = SpeechToSign(cfg, load_asr=False if args.no_asr else None)
    st = speech.status()
    print(f"  dictionary: {st['dictionary']['words']} words ({st['dictionary']['folder']})")
    print(f"  speech recognition: {'on ' + st['speech']['device'] if st['speech']['ok'] else st['speech']['message']}")
    print(f"  microphone: {'found' if st['microphone']['ok'] else st['microphone']['message']}")
    print("Loading Sign -> Text (sentence builder, sign model, camera, MediaPipe) ...", flush=True)
    camera = None if args.no_camera else (args.camera if args.camera is not None else "config")
    sign = SignToText(cfg, ROOT, camera_index=camera, lm=speech.predictor.lm if speech.predictor else None)
    if speech.predictor:
        print(f"  context prediction: {speech.predictor.lm.name} model"
              f"{' (' + speech.predict_message + ')' if speech.predict_message else ''}")
    ss = sign.status()
    print(f"  sign model: {ss['model']['name'] or ss['model']['message']}"
          f"{' (camera vocabulary ' + str(ss['model']['vocabulary']) + ' words)' if ss['model']['ok'] else ''}")
    print(f"  camera: {ss['message']}")
    print(f"  speech output: {'on' if ss['speech_output']['ok'] else ss['speech_output']['message']}")
    server = AppServer(cfg, ROOT, speech, sign)
    server.load_times["total"] = round(time.perf_counter() - started, 2)
    print(f"Loaded in {server.load_times['total']:.1f} s: {dict(speech.load_times, **sign.load_times)}", flush=True)
    url = server.start(open_browser=cfg["app"]["open_browser"] and not args.no_browser)
    print(f"SignBangla app: {url}  (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.close_all()


if __name__ == "__main__":
    main()
