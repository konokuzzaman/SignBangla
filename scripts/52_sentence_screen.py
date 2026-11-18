"""MODULE 3 screen: build a Bangla sentence from sign words in a browser page on this computer.

    python scripts/52_sentence_screen.py                (Words mode: click words, no camera needed)
    python scripts/52_sentence_screen.py --camera 0     (Camera mode too: webcam 0 -> Module 1 model -> sentence)

The page opens by itself (address printed below; 127.0.0.1 only). Press Enter or Ctrl+C here to stop.
Settings: configs/config.yaml -> module3.screen. Without a camera or model checkpoint the page still opens in Words
mode and says why camera mode is off.
"""
import argparse
import sys

import _path  # noqa: F401

from signbangla.sign2text.app_server import SentenceScreen
from signbangla.utils import load_config


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--camera", type=int, help="webcam index (0 = the built-in camera); also turns on Camera mode")
    ap.add_argument("--port", type=int, help="first port to try (default: module3.screen.port)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args()
    cfg = load_config(args.config)
    screen = SentenceScreen(cfg, camera_index=args.camera, port=args.port,
                            open_browser=False if args.no_browser else None)
    print(f"Sentence screen: {screen.url}")
    cam = screen.state.snapshot(0)
    if args.camera is not None:
        print(f"Camera: {cam['message']}")
    try:
        input("Press Enter here to stop.\n")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        screen.close()


if __name__ == "__main__":
    main()
