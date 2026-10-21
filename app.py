"""
app.py: Streamlit web app for Real-World Waste Detection.
Run with:  streamlit run app.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from detect import load_model, detect_waste, draw_detections, count_by_class, count_by_parent
from waste_info import (ALL_PARENTS, HAZARDOUS, HAZARD_DISCLAIMER, get_guidance, get_parent)

st.set_page_config(page_title="Waste Detection", page_icon="♻️", layout="wide")
st.title("♻️ Real-World Waste Detection and Classification")
st.write("Upload a photo of messy waste. The model finds each object, names it, and "
         "shows general disposal guidance.")

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("Settings")
    weights_path = st.text_input("Model file", value="best.pt")
    conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05,
                     help="Boxes with a score below this value are hidden. "
                          "Lower = finds more (but more false alarms).")
    st.markdown("**Box colours**  \n🟥 Hazardous  \n🟩 Biodegradable  \n🟦 Non-biodegradable / Recyclable")


@st.cache_resource
def get_model(path):
    """Load the YOLO model once and reuse it."""
    return load_model(path)


if not Path(weights_path).exists():
    st.error(f"Model file '{weights_path}' not found. Put your trained best.pt in this folder.")
    st.stop()
model = get_model(weights_path)

# ---------------- Image upload ----------------
uploaded = st.file_uploader("Upload a waste image (to try another image, simply upload a new one)",
                            type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("👆 Upload an image to begin.")
    st.stop()

pil_image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")   # fix phone-photo rotation
image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)           # OpenCV uses BGR

# ---------------- Detection ----------------
with st.spinner("Detecting waste objects..."):
    detections = detect_waste(model, image_bgr, conf=conf)

if len(detections) == 0:
    st.warning("No waste objects were detected. Try lowering the confidence threshold in the "
               "sidebar, or use a clearer photo. Remember: the model only knows the classes it was trained on.")
    st.image(pil_image, caption="Uploaded image")
    st.stop()

annotated_rgb = cv2.cvtColor(draw_detections(image_bgr, detections), cv2.COLOR_BGR2RGB)

# ---------------- Results ----------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Detection result")
    st.image(annotated_rgb, caption=f"{len(detections)} object(s) detected")

with right:
    st.subheader("Summary")
    st.metric("Total objects detected", len(detections))

    parent_counts = count_by_parent(detections)
    metric_cols = st.columns(len(ALL_PARENTS))
    for col, parent in zip(metric_cols, ALL_PARENTS):
        col.metric(parent, parent_counts.get(parent, 0))

    st.markdown("**Count per class**")
    class_counts = count_by_class(detections)
    st.bar_chart(pd.Series(class_counts))

# Hazardous caution
if parent_counts.get(HAZARDOUS, 0) > 0:
    st.warning("⚠️ **Possible hazardous waste detected.** " + HAZARD_DISCLAIMER)

# Table of every detection
st.subheader("All detections")
table = pd.DataFrame([{
    "Class": d["class_name"],
    "Parent category": d["parent"],
    "Confidence (%)": round(d["confidence"] * 100, 1),
} for d in sorted(detections, key=lambda d: -d["confidence"])])
st.dataframe(table, hide_index=True)

# Guidance, one expander per detected class
st.subheader("Disposal guidance")
st.caption("General guidance only. Local rules differ, so check with your local waste authority.")
for name in sorted(class_counts):
    with st.expander(f"{name}  ({get_parent(name)})  ×{class_counts[name]}"):
        st.write(get_guidance(name))

st.divider()
st.caption("Predictions come from a student-trained model and can be wrong or miss objects. "
           "The model only recognises the classes it was trained on.")
