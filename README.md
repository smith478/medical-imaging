# Medical imaging

This repo contains a set of tools for classification and detection tasks for medical imaging applications. 

## To run from docker

```bash
sudo docker run --gpus all --name tf_gpu -it --rm -p 8888:8888 -p 8501:8501 --entrypoint /bin/bash -v $(pwd):/medical-imaging -v /media/ssd_4TB:/data tensorflow/computervision:v3
```

## To run jupyter from the container

```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
```

## Examples
Lession classification
Lession detection - Class activation mapping
Radiograph position labeling

## TODO
Add examples of pulling pretrained huggingface models

