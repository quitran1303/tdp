# Instruction 

## General Files
- `model.py` contains classes and function to generate Mask R-CNN model
- `config.py` contains some hyper-parameters
- `utils.py` contains some help function which is more general
- `visualize.py` contains functions to draw images and visualize region proposals.
- `pycocotools` we use this tools to load mask.
- `evaluation_city.py` we use this to evaluate the AP to test our model

## Loading Dataset
To generate a all-in-one annotation file, we use `generate_annotations.py`.

## Training
The main code for training is `cityscape.py`. We use this file together with `mode.py`, `utils.py` and `config.py` to modify and train model.

## Evaluation
To evaluate out model, we use `demo` notebook to obtain bounding box, mask and classification outputs.
Also, we use 'evaluation_city.py' to get the APs, not only on the sinle image, but also on subset with any size of the whole validation set. The model_path indicates the model path to be loaded. 

## Visualization
We use `rpn_visualization` notebook to visualize region proposals and `inspect_weights` notebook to draw histogram of weights.

To visualize loss curves, we use **tensorboard**.

#reference, our codes modify the codes from https://github.com/matterport/Mask_RCNN .
