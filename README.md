## My personal background remover tool
This tool takes in an image (max 1500x1500 pixels) and removes the background using [DeepLabV3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/).\
The output is a png image with a transparent background available for download and the output size is the same as the input size.\
Useful for overlaying images in a PowerPoint presentation or your resume.

## Example

Original\
<img src="https://raw.githubusercontent.com/Chaxo/my_transparent_background/main/Example/original.jpg" width="250">
<img src="https://raw.githubusercontent.com/Chaxo/my_transparent_background/main/Example/original2.jpg" width="250">

Processed\
<img src="https://raw.githubusercontent.com/Chaxo/my_transparent_background/main/Example/processed.png" width="250">
<img src="https://raw.githubusercontent.com/Chaxo/my_transparent_background/main/Example/processed2.png" width="250">

## How to use:
The tool is hosted on Streamlit service.\
You can access it [here](https://share.streamlit.io/chaxo/background_remover/main/main.py).\
Simply upload an image and wait for it to be processed.
Once it is done processing, a download link and preview will appear.

## Why a maximum of 1500 pixel?
Unfortunately the free-tier of Streamlit can not handle heavy computations.\
To avoid crashing the tool, a limitation is set to 1500x1500 pixels.