import base64
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms
from io import BytesIO

# This cache might solve the memory issue on Streamlit
@st.cache(allow_output_mutation=True, hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    return model

def make_transparent_foreground(pic, mask):
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    a = np.ones(mask.shape, dtype='uint8') * 255
    alpha_im = cv2.merge([b, g, r, a], 4)
    bg = np.zeros(alpha_im.shape)
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)
    return foreground

def remove_background(model, input_file):
    input_image = Image.open(input_file)
    preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    with torch.no_grad():
        output = model(input_batch) ['out'] [0]
    
    output_predictions = output.argmax(0)
    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)
    foreground = make_transparent_foreground(input_image, bin_mask)
    return foreground, bin_mask

# Workaround download function so Streamlit page does not reload
def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href 

# Streamlit page layout
st.subheader("Welcome to my background remover tool")
st.markdown("Upload an image and it will remove the background and make it transparent.  \n" 
            "For more information, visit the [GitHub page](https://github.com/Chaxo/background_remover)")

# Upload image
uploaded_image = st.file_uploader("Limit 1500px image width and height" , type=["png","jpg","jpeg"])

# Once an image is uploaded
if uploaded_image is not None:
    # Uploaded image details
    uploaded_image_details = Image.open(uploaded_image)
    image_details = {"image_format":uploaded_image_details.format, "image_mode":uploaded_image_details.mode, "image_width":uploaded_image_details.width, "image_height":uploaded_image_details.height}
    st.write(image_details)

    # Handles wrong image mode exception
    if uploaded_image_details.mode == "RGBA":
        st.warning("The uploaded image file is of mode RGBA, where A indicates transparency.  \n This tool does not work with images containing transparency.  \n Please remove it or try another one")
        st.stop()

    # Handles too large image resolution where width or height exceeds 2000px
    if uploaded_image_details.width > 1500 or uploaded_image_details.height > 1500:
        st.warning("The width or height of the image exceeds 1500 pixels.  \n Please reduce the size and try again as Streamlit's free-tier computing power can not handle it")
        st.stop()

    # Preview the uploaded image
    st.write("Original:")
    st.image(uploaded_image, width=500)

    # Call the deeplabv3 and process uploaded image
    deeplab_model = load_model()
    foreground, bin_mask = remove_background(deeplab_model, uploaded_image)    
    processed_image = Image.fromarray(foreground)

    # Preview of processd image
    st.write("Processed:")
    st.image(processed_image, width=500)

    # Download button for the processed image
    st.markdown(get_image_download_link(processed_image,'output_image.png','Download'), unsafe_allow_html=True)