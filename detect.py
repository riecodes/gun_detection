"""Gun detection on a webcam or an RTSP CCTV stream.

    python detect.py                      # default webcam
    python detect.py 1                    # second webcam
    python detect.py rtsp://user:pass@host:554/stream1
    python detect.py rtsp://... 0.55      # higher confidence threshold
    python detect.py --selftest
"""
import os
import sys
import time

# CCTV over UDP drops packets and smears frames; force TCP before cv2 loads ffmpeg.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
from ultralytics import YOLO

# Weights: huggingface.co/cive202/weapon-detection-yolov8-cctv (best.pt -> gun_yolov8n.pt),
# a YOLOv8n fine-tune with two classes: person, weapon.
WEIGHTS = os.getenv("GUN_WEIGHTS", os.path.join(os.path.dirname(__file__), "gun_yolov8n.pt"))
ALERT_GAP = 5.0  # seconds between repeat alerts for a continuing sighting


def should_alert(hits, last, now):
    return bool(hits) and now - last > ALERT_GAP


def run(source=0, conf=0.4):
    model = YOLO(WEIGHTS)
    last = 0.0
    # stream=True yields per-frame results and handles webcam indexes and RTSP URLs alike.
    for result in model.predict(source=source, conf=conf, stream=True, verbose=False):
        hits = sorted({model.names[int(c)] for c in result.boxes.cls} - {"person"})
        now = time.time()
        if should_alert(hits, last, now):
            last = now
            print(f"[ALERT] {time.strftime('%H:%M:%S')} {', '.join(hits)}", flush=True)
        cv2.imshow("gun detection - esc quits", result.plot())
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def selftest():
    import numpy as np

    model = YOLO(WEIGHTS)
    assert "weapon" in model.names.values(), model.names
    result = model.predict(np.zeros((480, 640, 3), dtype="uint8"), verbose=False)[0]
    assert result.boxes is not None and len(result.boxes) == 0, "blank frame should detect nothing"
    assert should_alert(["weapon"], 0.0, 100.0)
    assert not should_alert([], 0.0, 100.0)
    assert not should_alert(["weapon"], 99.0, 100.0), "repeat alert should be debounced"
    print("selftest ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        selftest()
    else:
        source = args[0] if args else "0"
        run(int(source) if source.isdigit() else source, float(args[1]) if len(args) > 1 else 0.4)
