import base64
import cv2
from pyparsing import col
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
st.header("Welcome to my background remover")

# Upload image
uploaded_image = st.file_uploader("Limit 1080 x 1080 dimension per image due to Streamlit's limited processing capability" , type=["png","jpg","jpeg"])

# Once an image is uploaded
if uploaded_image is not None:
    # Uploaded image details
    image_details = {"filename":uploaded_image.name, "filetype":uploaded_image.type, "filesize":uploaded_image.size}
    st.write(image_details)

    # Preview the uploaded image
    img = Image.open(uploaded_image)
    st.image(uploaded_image, width=500)

    # Call the deeplabv3 and process uploaded image
    deeplab_model = load_model()
    foreground, bin_mask = remove_background(deeplab_model, uploaded_image)
    processed_image = Image.fromarray(foreground)

    # Preview of processd image
    st.image(processed_image, width=500)

    # Download button for the processed image
    st.markdown(get_image_download_link(processed_image,'output_image.png','Download'), unsafe_allow_html=True)

    
    
    
    
    
    
    
    
    
    ##  Old code for the download button. Problem was that it reloads the page after clicking. 
    ##  Streamlit is apparently working on adding a feature to not reload page.
    
            # # Convert to downloadable format
            # buf = BytesIO()
            # processed_image.save(buf, format="PNG")
            # byte_im = buf.getvalue()

            # # Download button
            # btn = st.download_button(
            #         label="Download image",
            #         data=byte_im,
            #         file_name="image.png",
            #         mime="image/png"
            #     )

