# Medical imaging

This repo contains a set of tools for classification and detection tasks for medical imaging applications. 

## To run from docker

The following docker image is built from the base tensorflow image with the addition of the Python libraries in `requirements.txt`.

```bash
docker run --gpus all --name tf_gpu -it --rm -p 8888:8888 -p 8501:8501 --entrypoint /bin/bash -w /medical-imaging -v $(pwd):/medical-imaging -v /media/ssd_4TB:/data tensorflow/computervision:v3
```

## To run jupyter from the container

```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
```

## To run the streamlit app

```bash
streamlit run app.py
```

## Dataset
We will be training our model on the Stanford [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) which contains over 200k of radiologist labeled x-ray images.

## Examples
- Lession classification (notebook and streamlit app)
    - Notebook: `./notebooks/Model_Training.ipynb`
Lession detection - Class activation mapping (notebook and streamlit app)
    - Notebook: `./notebooks/class_activation_map_visualization.ipynb`
    - Streamlit app: `./app.py`
Radiograph position labeling (notebook and streamlit app)
    - Notebook: `./notebooks/radiology_position_labeling.ipynb`
    - Streamlit app: `./app_image_clean.py`

## TODO
- Add examples of pulling pretrained huggingface models
- Look at timm models
- Look at Keras 3

