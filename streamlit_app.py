from pathlib import Path
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="PlantDoc | Leaf Scanner", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#071b15,#102b20,#071411);color:#f4fff5}
.block-container {max-width:1100px;padding-top:2rem}
h1,h2,h3 {color:#eaffec}
[data-testid="stFileUploader"] section {background:#12271d;border:1px dashed #72b77d}
.stButton button {background:#45a867;color:white;border:0;border-radius:10px}
</style>
""", unsafe_allow_html=True)

st.title("🌿 PlantDoc Leaf Health Scanner")
st.write("Upload a plant-leaf photo or take one with your camera to detect possible disease labels.")
st.caption("Student project demo · YOLO11n · Predictions may be incorrect.")

with st.sidebar:
    st.header("Scan settings")
    threshold = st.slider("Detection threshold", 0.10, 0.80, 0.25, 0.05)
    st.caption("Lower values show more possible detections.")

upload_tab, camera_tab = st.tabs(["📁 Upload photo", "📷 Take photo"])
with upload_tab:
    photo = st.file_uploader("Choose a clear leaf photo", type=["jpg", "jpeg", "png"])
with camera_tab:
    camera_photo = st.camera_input("Take a leaf photo")
if camera_photo is not None:
    photo = camera_photo

if photo is not None:
    image = Image.open(photo).convert("RGB")
    model_path = Path(__file__).parent / "best.pt"

    if not model_path.exists():
        st.error("Model file missing. Upload best.pt to this same GitHub repository.")
    else:
        @st.cache_resource
        def load_model(path):
            return YOLO(path)

        with st.spinner("Scanning the leaf…"):
            result = load_model(str(model_path)).predict(
                image, imgsz=512, conf=threshold, verbose=False
            )[0]

        left, right = st.columns([1.2, 0.8])
        with left:
            st.subheader("Scanned photo")
            st.image(result.plot()[:, :, ::-1], use_container_width=True)
        with right:
            st.subheader("Predictions")
            if len(result.boxes) == 0:
                st.warning("No detection found. Try a closer, clearer photo.")
            for box in result.boxes:
                label = result.names[int(box.cls[0])].replace("_", " ")
                score = float(box.conf[0])
                st.success(f"{label}  ·  {score:.1%} model score")
            st.caption("A model score is confidence for that detection, not overall accuracy.")
else:
    st.info("Start by uploading a photo or using the camera tab.")
