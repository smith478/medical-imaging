import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
import matplotlib.cm as cm

model_path = os.path.join('/models', 'pretrain_model_ConvNeXtBase_w_ClssWgt_01-0.3616')
save_path = os.path.join('heatmap_tmp', 'heatmap_cam.jpg')
img_size = (640, 640)
disease_name = 'Pulmonary Nodules'
last_conv_layer_name = 'block14_sepconv2'
classifier_layer_names = ['local_avg_pool', 'flatten', 'prediction']

#@st.cache_resource
def model_load():
    model = tf.keras.models.load_model(model_path)
    return model


def main():
    st.set_page_config(layout="wide")

    st.markdown("<h1>Lesion Detection</h1>", unsafe_allow_html=True)
    st.text("""
    Select a radiograph to see both the model's predicted likelihood 
    of enlarged cardiomediastinum, cardiomegaly, lung opacity, lung lesion, edema,
    consolidation, pneumonia, atelectasis, pneumothorax, pleural effusion, pleural other,
    fracture, support devices, and the heatmap of the most concerning regions of the 
    radiograph.
    """)
    model = model_load()

    alpha = st.sidebar.slider('Heatmap opacity adjust', min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    Image = st.file_uploader('Upload your radiograph here', type=['jpg', 'jpeg', 'png'])

    if Image is not None:
        col1, col2 = st.columns(2)
        img = Image.read()
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize_with_pad(img, target_height=750, target_width=750).numpy()
        img = img.astype(np.float32) / 255.

        with col1:
            st.image(img)

        img = np.expand_dims(img, axis=0)
        img_pred = model.predict(img)

        heatmap = make_gradcam_heatmap(img_array=img, model=model, last_conv_layer_name=last_conv_layer_name,
                                       classifier_layer_names=classifier_layer_names)

        # We rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = np.squeeze(jet_heatmap)
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize(img_size)
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        img = np.squeeze(img)
        img *= 255.
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = superimposed_img.astype(np.float32) / ((1 + alpha) * 255.)

        with col2:
            st.image(superimposed_img)

        probability = 'Probability of {} is {:.1f}%'.format(disease_name, 100*img_pred[0][1])
        st.text(probability)


if __name__ == '__main__':
    main()