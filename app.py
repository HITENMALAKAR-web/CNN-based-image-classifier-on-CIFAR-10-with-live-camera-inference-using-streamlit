import streamlit as st
import PIL
from PIL import Image
from src.data_loader import class_names

from src.CNN_Architecture import Hnet
from src.pedict import load_image,get_modal
import torch
st.set_page_config(page_title="IMAGE CLASSIFIER",page_icon=	':telescope:')

@st.cache_resource
def build_model():
    return get_modal()
net = build_model()

st.title("CIFAR-10 Image Classifier")
st.write(" this image classifier can identify images of plane,car,bird,cat,deer,dog,frog,horse,ship,truck . you can give input image of jpg,jpeg,png format")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    if st.button("Run"):
        image = load_image(uploaded_file)
        with torch.no_grad():
            output = net(image)
            a, predicted = torch.max(output, 1)
            st.write(f"prediction: {class_names[predicted.item()]}")


