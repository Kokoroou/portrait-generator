import streamlit as st
from PIL import Image
from src.controller.GANController import *

st.markdown("# Portrait Generate")

expander1 = st.expander("Upload photo", expanded=True)

image_file = expander1.file_uploader("", type=["png", "jpg", "jpeg"])

if image_file is not None:
    image = Image.open(image_file)
    expander1.image(image, caption="Your photo")

    expander2 = st.expander("Choose style", expanded=False)

    style_index = expander2.selectbox(label="Style", options=list(range(0, 17)), index=0)
    expander2.image(image, caption="Your photo")

    expander3 = st.expander("Change color", expanded=False)

    structure_index = expander3.slider("Structure", 0, 6, 0)
    color_index = expander3.slider("Color", 0, 6, 0)

    expander3.image(image, caption="Your photo")
    expander3.download_button(label="Save", data=image_to_byte(image), file_name="DualStyleGAN_Image.png", mime="image/png")

