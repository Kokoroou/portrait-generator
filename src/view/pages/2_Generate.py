import streamlit as st
from PIL import Image
from src.controller.GANController import *
from src.model.gen_style import *
from src.model.gen_color import *
import numpy as np

st.markdown("# Portrait Generate")

expander1 = st.expander("Upload photo", expanded=True)

image_file = expander1.file_uploader("", type=["png", "jpg", "jpeg"])

if image_file is not None:
    image0 = Image.open(image_file)
    gene0 = image_to_gene(image0)

    expander1.image(image0, caption="Your photo")

    expander2 = st.expander("Choose style", expanded=False)

    style = expander2.selectbox(label="Style",
                                options=["Random"] + [style_name.capitalize() for style_name in styles],
                                index=0)
    style_index = -1
    if style == "Random":
        style_index = -1
    elif style.lower() in styles:
        style_index = styles.index(style.lower())

    gene1, info = gen_style(gene0, style_index)
    image1 = gene_to_image(gene1)
    expander2.image(image1, caption="New style")

    # new_colors = np.full((MAX_STRUCTURE_CODE, MAX_COLOR_CODE), 0)
    # for i in range(MAX_STRUCTURE_CODE):
    #     for j in range(MAX_COLOR_CODE):
    #         new_colors[i][j] = gen_color(gene1, i, j, info)

    expander3 = st.expander("Change color", expanded=False)

    structure_index = expander3.slider("Structure", 0, MAX_STRUCTURE_CODE, 3)
    color_index = expander3.slider("Color", 0, MAX_COLOR_CODE, 3)

    # gene2 = new_colors[structure_index][color_index]
    gene2 = gen_color(gene1, structure_index, color_index, info)
    image2 = gene_to_image(gene2)

    expander3.image(image1, caption="Your photo")
    expander3.download_button(label="Save", data=image_to_byte(image2), file_name="DualStyleGAN_Image.png",
                              mime="image/png")
