#!/usr/bin/env python3
import json
import statistics
import sys

try:
    from fast_alpr import ALPR
except Exception as exc:
    print(json.dumps({"plate": "Unknown", "confidence": 0.0, "error": f"import: {exc}"}))
    sys.exit(0)

_alpr = None

def get_alpr():
    global _alpr
    if _alpr is None:
        _alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            detector_conf_thresh=0.35,
            ocr_model="cct-xs-v2-global-model",
            ocr_device="cpu",
        )
    return _alpr

def confidence_value(ocr):
    if ocr is None or getattr(ocr, "confidence", None) is None:
        return 0.0
    c = ocr.confidence
    if isinstance(c, (list, tuple)):
        return float(statistics.mean(c)) if c else 0.0
    return float(c)

def recognize(image_path):
    try:
        results = get_alpr().predict(image_path) or []
        best_text = "Unknown"
        best_conf = 0.0
        for r in results:
            ocr = getattr(r, "ocr", None)
            text = (getattr(ocr, "text", "") or "").strip().upper().replace(" ", "")
            conf = confidence_value(ocr)
            if text and conf >= best_conf:
                best_text = text
                best_conf = conf
        return {"plate": best_text or "Unknown", "confidence": best_conf}
    except Exception as exc:
        return {"plate": "Unknown", "confidence": 0.0, "error": str(exc)}

def worker():
    get_alpr()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        image_path = line.strip()
        if not image_path:
            continue
        if image_path == "__quit__":
            break
        print(json.dumps(recognize(image_path), separators=(",", ":")), flush=True)

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        worker()
        return
    if len(sys.argv) != 2:
        print(json.dumps({"plate": "Unknown", "confidence": 0.0, "error": "usage: parkcam_alpr.py image.jpg"}))
        return
    print(json.dumps(recognize(sys.argv[1]), separators=(",", ":")))

if __name__ == "__main__":
    main()
