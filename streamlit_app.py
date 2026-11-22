from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO


st.set_page_config(
    page_title="PlantDoc Vision | Plant Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "best.pt"

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
      .stApp { background: #f5f4ec; color: #18392b; }
      [data-testid="stHeader"] { background: rgba(245,244,236,.92); }
      .block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem; }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #163b2b; }
      .hero { background: linear-gradient(115deg,#123c2d,#1b563b); border-radius: 24px; padding: 2.2rem 2.5rem; margin-bottom: 1.4rem; color: #f5f6eb; }
      .hero-kicker { color:#b9d7a9; text-transform:uppercase; letter-spacing:.16em; font-weight:700; font-size:.76rem; }
      .hero h1 { color:#fffef5; font-size:clamp(2.2rem,5vw,3.35rem); line-height:1.08; margin:.55rem 0 .65rem; }
      .hero p { color:#d4e2d2; font-size:1.06rem; max-width:720px; margin:0; }
      .subtle { color:#5a7264; }
      div[data-testid="stFileUploader"] section { border:1px dashed #73977c; background:#fffef8; border-radius:16px; }
      div[data-testid="stCameraInput"] { border-radius:16px; overflow:hidden; }
      div.stButton > button, div[data-testid="stDownloadButton"] button { border-radius:10px; font-weight:700; min-height:2.8rem; }
      div.stButton > button[kind="primary"] { background:#2e754b; border-color:#2e754b; }
      [data-testid="stMetric"] { background:#fffef8; border:1px solid #e1e6d9; border-radius:14px; padding:1rem; }
      [data-testid="stVerticalBlockBorderWrapper"] { border-color:#dce3d8 !important; border-radius:16px !important; background:#fffef8; }
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">PlantDoc Vision</div>
      <h1>A clearer look at leaf health</h1>
      <p>Add a leaf photo to see the regions and PlantDoc labels detected by the model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Scan settings")
    confidence = st.slider(
        "Minimum detection confidence",
        0.10,
        0.80,
        0.25,
        0.05,
        help="This filters weaker boxes; it does not change the model's trained accuracy.",
    )
    st.caption("Lower values show more possible detections. Higher values hide weaker ones.")

st.markdown("#### Add a photo")
upload_tab, camera_tab = st.tabs(["Upload photo", "Use camera"])
with upload_tab:
    uploaded = st.file_uploader(
        "Choose one clear leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
with camera_tab:
    camera_photo = st.camera_input("Take a photo of a leaf", label_visibility="collapsed")

photo = camera_photo if camera_photo is not None else uploaded


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def draw_clean_boxes(source_image: Image.Image, result) -> Image.Image:
    marked = source_image.copy()
    draw = ImageDraw.Draw(marked)
    line_width = max(3, round(min(marked.size) / 220))
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        label = result.names[int(box.cls[0])].replace("_", " ")
        score = float(box.conf[0])
        tag = f"{label}  {score:.0%}"
        draw.rectangle((x1, y1, x2, y2), outline="#1b8a50", width=line_width)
        text_box = draw.textbbox((0, 0), tag)
        tag_w = text_box[2] - text_box[0] + 18
        tag_h = text_box[3] - text_box[1] + 12
        tag_y = max(0, y1 - tag_h)
        draw.rounded_rectangle((x1, tag_y, x1 + tag_w, tag_y + tag_h), radius=6, fill="#1b8a50")
        draw.text((x1 + 9, tag_y + 5), tag, fill="white")
    return marked


if photo is None:
    st.markdown(
        '<p class="subtle">Use a well-lit photo where a leaf fills most of the frame. The detector works best when the leaf is in focus.</p>',
        unsafe_allow_html=True,
    )
else:
    image = Image.open(photo).convert("RGB")
    preview, action = st.columns([1, 0.32], vertical_alignment="center")
    with preview:
        st.image(image, caption="Photo selected", use_container_width=True)
    with action:
        st.write("")
        st.write("")
        analyze = st.button("Analyze photo", type="primary", use_container_width=True)

    if analyze:
        if not MODEL_PATH.exists():
            st.error("The model file best.pt is missing from the app repository.")
        else:
            try:
                with st.spinner("Checking the photo…"):
                    model = load_model(str(MODEL_PATH))
                    result = model.predict(image, imgsz=512, conf=confidence, verbose=False)[0]
                    marked = draw_clean_boxes(image, result)

                st.markdown("---")
                st.markdown("#### Scan results")
                if len(result.boxes) == 0:
                    st.image(image, caption="No region passed the selected threshold", use_container_width=True)
                    st.info("Try a clearer or closer photo, or lower the confidence setting slightly.")
                else:
                    image_col, result_col = st.columns([1.2, 0.8], gap="large")
                    with image_col:
                        st.image(marked, caption="Detected leaf regions", use_container_width=True)
                        buffer = BytesIO()
                        marked.save(buffer, format="JPEG", quality=92)
                        st.download_button(
                            "Download marked photo",
                            data=buffer.getvalue(),
                            file_name="plantdoc_scan_result.jpg",
                            mime="image/jpeg",
                        )
                    with result_col:
                        st.metric("Regions detected", len(result.boxes))
                        for box in result.boxes:
                            label = result.names[int(box.cls[0])].replace("_", " ")
                            score = float(box.conf[0])
                            st.container(border=True).markdown(
                                f"**{label}**  \nModel score: **{score:.1%}**"
                            )
                        st.caption("A model score is confidence for one detection, not overall test accuracy.")
            except Exception as error:
                st.error(f"The photo could not be analyzed: {error}")

st.markdown("---")
st.caption(
    "PlantDoc research project. Results can be wrong, especially on field photos or unfamiliar conditions. "
    "Use this tool for learning and demonstration, not for treatment decisions."
)

