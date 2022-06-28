import streamlit as st

from src.model.gen_style import *

st.markdown("# Portrait Generate")

# Step 1: Upload photo
expander1 = st.expander("Upload photo", expanded=True)
image_file = expander1.file_uploader("", type=["png", "jpg", "jpeg"])

if image_file is not None:
    generator = StyleGenerator()

    # Show image
    image0 = Image.open(image_file)
    expander1.image(image0, caption="Your photo")

    # Change image to gene for next step
    gene0 = image_to_gene(image0)

    # Step 2: Modify parameter for generator
    expander2 = st.expander("Choose style", expanded=True)
    style = expander2.selectbox(label="Style", options=["Random"] + [style_name.capitalize() for style_name in styles],
                                index=6)
    structure_rate = expander2.slider(label="Structure", min_value=0.0, max_value=1.0, value=0.5, step=0.25)
    color_rate = expander2.slider(label="Color", min_value=0.0, max_value=1.0, value=0.0, step=0.25)

    # Digitalizing choice for generator
    style_index = -1
    if style == "Random":
        style_index = -1
    elif style.lower() in styles:
        style_index = styles.index(style.lower())

    # Modify generator
    generator.modified_generator(style_index=style_index, structure_rate=structure_rate, color_rate=color_rate)

    # Generate new style
    generator.add_gene(gene0)
    gene1 = generator.generate()

    # Show new image
    image1 = gene_to_image(gene1)
    expander2.image(image1, caption="New style")

    # Download new image
    expander2.download_button(label="Save", data=image_to_byte(image1), file_name="DualStyleGAN_Image.png",
                              mime="image/png")
