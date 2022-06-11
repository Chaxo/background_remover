## My personal background remover tool
This tool takes in an image and removes the background using [DeepLabV3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/).\
The output is a png image with a transparent background available for download.

**Example**

Original\
<img src="https://raw.githubusercontent.com/Chaxo/my_transparent_background/main/Example/original.jpg" width="250">

Processed\
<img src="https://raw.githubusercontent.com/Chaxo/my_transparent_background/main/Example/processed.png" width="250">

## How to use:
The tool is hosted on Streamlit service.\
You can access it [here](https://share.streamlit.io/chaxo/background_remover/main/main.py).\
Simply upload an image and wait for it to be processed.

## To-do list:
- [] Handle exception (RGBA)
- [] Limit image dimension
- [] Add introduction/instructions
- [] Improve UI
