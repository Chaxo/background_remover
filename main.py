import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


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


deeplab_model = load_model()
foreground, bin_mask = remove_background(deeplab_model, 'C:/Users/Chaxo/Documents/my_transparent_background/input_pics/dog.jpg')
plt.imshow(foreground)
Image.fromarray(foreground).save("C:/Users/Chaxo/Documents/my_transparent_background/output_pics/dog.png")