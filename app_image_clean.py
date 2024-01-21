import numpy as np
import os
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")
# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("rect", "point", "freedraw", "line", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 1)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

blackout_top_left_corner = st.sidebar.checkbox("Blackout top left corner", True)
blackout_middle_borders = st.sidebar.checkbox("Blackout middle borders", False)
blackout_borders = st.sidebar.checkbox("Blackout borders", True)

realtime_update = st.sidebar.checkbox("Update in realtime", True)
delete_base_image_after_save = st.sidebar.checkbox("Delete uploaded image", True)

save_dir = st.sidebar.text_input(label='save directory', value=SAVE_DIR)
source_dir = st.sidebar.text_input(label='source directory', value=SOURCE_DIR)

# Upload image here, and feed it into canvas_result. Use the height and width to populate canvas size
img_upload = st.file_uploader('Upload an image.', type=['jpg', 'jpeg', 'png'])

if img_upload is not None:
    path_in = img_upload.name
    filename = os.path.basename(path_in)

    img = Image.open(img_upload)
    img_shape = img.size

    if blackout_top_left_corner or blackout_middle_borders or blackout_borders:
        draw = ImageDraw.Draw(img, "RGBA")

    if blackout_top_left_corner:
        draw.rectangle(((0, 0), (1500, 850)), fill=(0, 0, 0, 255))

    if blackout_middle_borders:
        mid_height = img_shape[1] // 2
        mid_width = img_shape[0] // 2
        # Left
        draw.rectangle(((0, mid_height - 100), (200, mid_height + 100)), fill=(0, 0, 0, 255))
        # Right
        draw.rectangle(((img_shape[0] - 200, mid_height - 100), (img_shape[0], mid_height + 100)), fill=(0, 0, 0, 255))
        # Top
        draw.rectangle(((mid_width - 100, 0), (mid_width + 100, 200)), fill=(0, 0, 0, 255))
        # Bottom
        draw.rectangle(((mid_width - 100, img_shape[1] - 200), (mid_width + 100, img_shape[1])), fill=(0, 0, 0, 255))

    if blackout_borders:
        # Left
        draw.rectangle(((0, 0), (200, img_shape[1])), fill=(0, 0, 0, 255))
        # Right
        draw.rectangle(((img_shape[0] - 200, 0), (img_shape[0], img_shape[1])), fill=(0, 0, 0, 255))
        # Top
        draw.rectangle(((0, 0), (img_shape[0], 200)), fill=(0, 0, 0, 255))
        # Bottom
        draw.rectangle(((0, img_shape[1] - 200), (img_shape[0], img_shape[1])), fill=(0, 0, 0, 255))

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 255)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color="rgba(0, 0, 0, 255)",
        background_color=bg_color,
        background_image=img,
        update_streamlit=realtime_update,
        height=img_shape[1] // 3,
        width=img_shape[0] // 3,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    with st.form('Download image'):
        save_image = st.form_submit_button('Save Image')

        if save_image:
            img = img.resize((img_shape[0] // 3, img_shape[1] // 3))
            save_img = Image.fromarray(canvas_result.image_data, 'RGBA')
            img.paste(save_img, (0, 0), save_img)
            img.save(os.path.join(save_dir, filename))

            if delete_base_image_after_save:
                os.remove(os.path.join(source_dir, path_in))

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)

    with st.form('Delete image'):
        delete_image = st.form_submit_button('Delete Image without Save')

        if delete_image:
            os.remove(os.path.join(source_dir, path_in))