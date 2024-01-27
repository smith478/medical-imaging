import streamlit as st
import tensorflow as tf
from keras.applications.convnext import LayerScale
from notebooks.grad_cam import make_gradcam_heatmap
import numpy as np
import os
from tensorflow import keras
import matplotlib.cm as cm

model_path = os.path.join('models', 'pretrain_model_ConvNeXtSmall_w_ClssWgt_03-0.5216.h5')
save_path = os.path.join('heatmap_tmp', 'heatmap_cam.jpg')
img_size = (640, 640)
disease_name = 'Pulmonary Nodules'
last_conv_layer_name = 'block14_sepconv2'
classifier_layer_names = ['local_avg_pool', 'flatten', 'prediction']

@st.cache_resource
def model_load():
    model = tf.keras.models.load_model(model_path, custom_objects={'LayerScale': LayerScale})
    return model

def update_inference_required():
    st.session_state['inference_required'] = True

def main():
    st.set_page_config(layout="wide")

    st.markdown("<h1>Lesion Detection</h1>", unsafe_allow_html=True)
    st.text("""
    Select a chest radiograph to see both the model's predicted likelihood of: 
    * Enlarged Cardiomediastinum 
    * Cardiomegaly 
    * Lung Opacity 
    * Lung Lesion 
    * Edema
    * Consolidation 
    * Pneumonia 
    * Atelectasis 
    * Pneumothorax 
    * Pleural Effusion 
    * Pleural Other
    * Fracture 
    * Support Devices
    
    🔍 Use the sidebar drop down to select the predicted abnormality to visualize (via class activation mapping). 
    ─ ● ─ Use the opacity slider to adjust the heatmap opacity.
    """)
    model = model_load()

    model_classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                     'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                     'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                     'Support Devices']

    if 'inference_required' not in st.session_state:
        st.session_state['inference_required'] = True

    alpha = st.sidebar.slider('Heatmap opacity adjust', min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    grad_cam_class = st.sidebar.selectbox('Select a disease to show the heatmap for:', model_classes, on_change=update_inference_required)
    grad_cam_index = model_classes.index(grad_cam_class)

    Image = st.file_uploader('Upload your radiograph here', type=['jpg', 'jpeg', 'png'], on_change=update_inference_required)

    if Image is not None:
        col1, col2 = st.columns(2)
        img = Image.read()
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize_with_pad(img, target_height=img_size[0], target_width=img_size[1]).numpy()
        img = img.astype(np.float32) / 255.

        with col1:
            st.image(img)

        img = np.expand_dims(img, axis=0)

        if st.session_state['inference_required']:
            img_pred = model.predict(img)
            heatmap = make_gradcam_heatmap(img_array=img, 
                                           model=model, 
                                           last_conv_layer_name=last_conv_layer_name,
                                           classifier_layer_names=classifier_layer_names,
                                           class_index=grad_cam_index)

            # We rescale heatmap to a range 0-255
            heatmap = np.uint8(255 * heatmap)

            st.session_state['img_pred'] = img_pred
            st.session_state['heatmap'] = heatmap
            st.session_state['inference_required'] = False

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[st.session_state['heatmap']]

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

        for class_name, class_prob in zip(model_classes, st.session_state['img_pred'][0]):
            st.text('{}: {:.1f}%'.format(class_name, 100*class_prob))
        # probability = 'Probability of {} is {:.1f}%'.format(disease_name, 100*st.session_state['img_pred'][0][1])
        # st.text(probability)


if __name__ == '__main__':
    main()