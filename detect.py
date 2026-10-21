"""
detect.py
Loads the trained YOLO model, finds waste objects in an image, draws boxes, and counts objects.
Can be imported by app.py or run from the command line:
    python detect.py --weights best.pt --image photo.jpg --out result.jpg
"""
import argparse

import cv2
from ultralytics import YOLO

from waste_info import PARENT_COLOR_BGR, get_parent


def load_model(weights_path):
    """Load the trained YOLO model from a .pt file."""
    return YOLO(str(weights_path))


def detect_waste(model, image_bgr, conf=0.25, iou=0.5, imgsz=640):
    """
    Run detection on ONE image (an OpenCV/BGR numpy array).
    conf = confidence threshold (hide boxes below this score)
    iou  = overlap limit used to remove duplicate boxes on the same object (non-max suppression)
    Returns a list of dictionaries, one per detected object.
    """
    result = model.predict(source=image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]   # corners of the box in pixels
        detections.append({
            "class_name": class_name,
            "parent": get_parent(class_name),
            "confidence": float(box.conf[0]),
            "box": (x1, y1, x2, y2),
        })
    return detections


def draw_detections(image_bgr, detections):
    """Return a COPY of the image with a coloured box + 'class 0.87' label on every detection."""
    out = image_bgr.copy()
    scale = max(out.shape[:2]) / 1000.0                  # make lines/text readable on any image size
    thickness = max(2, int(3 * scale))
    font_scale = max(0.5, 0.7 * scale)

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = PARENT_COLOR_BGR.get(d["parent"], (128, 128, 128))   # colour = parent category
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(0, y1 - th - baseline)
        cv2.rectangle(out, (x1, top), (x1 + tw, top + th + baseline), color, -1)      # label background
        cv2.putText(out, label, (x1, top + th), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), max(1, thickness // 2))
    return out


def count_by_class(detections):
    """{'plastic_bottle': 3, 'metal_can': 1, ...}"""
    counts = {}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return counts


def count_by_parent(detections):
    """{'Hazardous': 1, 'Non-biodegradable / Recyclable': 4, ...}"""
    counts = {}
    for d in detections:
        counts[d["parent"]] = counts.get(d["parent"], 0) + 1
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect waste in one image")
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="result.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model = load_model(args.weights)
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    found = detect_waste(model, image, conf=args.conf)
    cv2.imwrite(args.out, draw_detections(image, found))

    if not found:
        print("No waste objects detected.")
    else:
        print(f"Detected {len(found)} object(s):")
        for d in found:
            print(f'  {d["class_name"]:22s} {d["confidence"]:.2f}  [{d["parent"]}]')
        print("Counts per class :", count_by_class(found))
        print("Counts per parent:", count_by_parent(found))
    print("Saved annotated image to", args.out)
