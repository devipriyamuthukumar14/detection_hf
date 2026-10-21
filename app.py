"""
app.py: Streamlit web app for Real-World Waste Detection.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from detect import (
    load_model,
    detect_waste,
    draw_detections,
    count_by_class,
    count_by_parent,
)

from waste_info import get_parent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real-World Waste Detection",
    page_icon="♻️",
    layout="wide",
)

st.title("♻️ Real-World Waste Detection and Classification")

st.write(
    "Upload an image containing waste. The AI model detects individual "
    "waste objects, identifies their class, and assigns each object to "
    "a parent waste category."
)

APP_DIR = Path(__file__).parent


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    weights_path = st.text_input(
        "Model file",
        value=str(APP_DIR / "best.pt"),
    )

    conf = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Lower values detect more objects but may produce more false detections.",
    )

    st.markdown("---")

    st.markdown(
        """
        ### Waste Categories

        🟥 **Hazardous**

        🟩 **Biodegradable**

        🟦 **Non-biodegradable / Recyclable**
        """
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model(path):
    """Load the YOLO model once."""
    return load_model(path)


if not Path(weights_path).exists():

    st.error(
        f"Model file not found:\n\n"
        f"`{weights_path}`\n\n"
        "Please place your trained `best.pt` file in the project folder."
    )

    st.stop()


try:

    model = get_model(weights_path)

except Exception as e:

    st.error(f"Could not load the YOLO model: {e}")

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "📷 Upload a waste image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing one or more waste objects.",
)


if uploaded is None:

    st.info("👆 Upload an image to start waste detection.")

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

try:

    pil_image = ImageOps.exif_transpose(
        Image.open(uploaded)
    ).convert("RGB")

    image_bgr = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR,
    )

except Exception as e:

    st.error(f"Could not read the uploaded image: {e}")

    st.stop()


# ============================================================
# DISPLAY ORIGINAL IMAGE
# ============================================================

with st.expander("📷 View Uploaded Image"):

    st.image(
        pil_image,
        caption="Uploaded Image",
        use_container_width=True,
    )


# ============================================================
# RUN DETECTION
# ============================================================

with st.spinner("🔍 Detecting waste objects..."):

    try:

        detections = detect_waste(
            model,
            image_bgr,
            conf=conf,
        )

    except TypeError:

        try:

            detections = detect_waste(
                model,
                image_bgr,
                conf,
            )

        except Exception as e:

            st.error(f"Detection failed: {e}")

            st.stop()

    except Exception as e:

        st.error(f"Detection failed: {e}")

        st.stop()


# ============================================================
# DRAW DETECTION BOXES
# ============================================================

try:

    annotated_bgr = draw_detections(
        image_bgr.copy(),
        detections,
    )

    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB,
    )

except Exception as e:

    st.error(f"Could not draw detection boxes: {e}")

    st.stop()


# ============================================================
# DETECTION RESULT
# ============================================================

st.subheader("🔎 Detection Result")

st.image(
    annotated_rgb,
    caption="Detected Waste Objects",
    use_container_width=True,
)


# ============================================================
# NO DETECTIONS
# ============================================================

if not detections:

    st.warning(
        "No waste objects were detected."
    )

    st.info(
        "Try lowering the confidence threshold or upload another image."
    )

    st.stop()


# ============================================================
# GET COUNTS
# ============================================================

try:

    class_counts = count_by_class(detections)

except Exception:

    class_counts = {}


try:

    parent_counts = count_by_parent(detections)

except Exception:

    parent_counts = {}


# ============================================================
# SUMMARY
# ============================================================

st.subheader("📊 Waste Summary")

total_objects = len(detections)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Objects",
        total_objects,
    )


with col2:

    st.metric(
        "Hazardous",
        parent_counts.get("Hazardous", 0),
    )


with col3:

    st.metric(
        "Biodegradable",
        parent_counts.get("Biodegradable", 0),
    )


with col4:

    st.metric(
        "Non-biodegradable / Recyclable",
        parent_counts.get(
            "Non-biodegradable / Recyclable",
            parent_counts.get("Non-biodegradable", 0),
        ),
    )


# ============================================================
# CLASS-WISE COUNTS
# ============================================================

st.subheader("🏷️ Detected Waste Classes")

class_rows = []

for class_name, count in class_counts.items():

    try:

        parent = get_parent(class_name)

    except Exception:

        parent = "Unknown"

    class_rows.append(
        {
            "Waste Class": class_name,
            "Count": count,
            "Category": parent,
        }
    )


if class_rows:

    class_df = pd.DataFrame(class_rows)

    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info("No class information available.")


# ============================================================
# INDIVIDUAL DETECTION DETAILS
# ============================================================

st.subheader("🔍 Individual Detection Details")

detail_rows = []


for detection in detections:

    if isinstance(detection, dict):

        class_name = detection.get(
            "class_name",
            detection.get(
                "name",
                detection.get(
                    "class",
                    "Unknown",
                ),
            ),
        )

        confidence = detection.get(
            "confidence",
            detection.get(
                "conf",
                0,
            ),
        )

    else:

        class_name = str(detection)

        confidence = 0


    try:

        parent = get_parent(class_name)

    except Exception:

        parent = "Unknown"


    try:

        confidence_value = float(confidence)

    except Exception:

        confidence_value = 0.0


    detail_rows.append(
        {
            "Detected Object": class_name,
            "Category": parent,
            "Confidence": f"{confidence_value:.1%}",
        }
    )


if detail_rows:

    details_df = pd.DataFrame(detail_rows)

    st.dataframe(
        details_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CATEGORY BREAKDOWN
# ============================================================

st.subheader("📈 Category Breakdown")

if parent_counts:

    category_rows = []

    for category, count in parent_counts.items():

        category_rows.append(
            {
                "Category": category,
                "Number of Objects": count,
            }
        )

    category_df = pd.DataFrame(category_rows)

    st.dataframe(
        category_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "This application uses YOLO-based object detection to identify "
    "waste objects in images. Predictions are based on the classes "
    "learned during model training and may be incorrect for unclear "
    "or previously unseen objects."
)
