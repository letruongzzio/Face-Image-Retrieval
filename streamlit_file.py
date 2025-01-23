import streamlit as st
from itertools import cycle
import os
import time
import PIL.Image
import sys
import cv2
PARENT_DIRNAME = os.path.expanduser("~/image-processing-project/")
sys.path.append(os.path.join(PARENT_DIRNAME, "lib/"))
from processing import crop_face
import torch

import json
import requests
from streamlit_lottie import st_lottie

def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def save_image(img):
    img_folder = os.path.join(PARENT_DIRNAME, "backend_storage")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    img_path = f"{img_folder}/{timestamp}.png"

    with open(img_path, 'wb') as f:
        f.write(img.getvalue())
        st.success(f"Image saved to {img_path}")
    return img_path



def process_image(image_path):
    image = cv2.imread(image_path)
    state, cropped_image = crop_face(image, 1.12, 16)
    if state == "one":
        cv2.imwrite(image_path, cropped_image)
        return img_path
    else:
        return None


# Design the layout of the Streamlit app
st.set_page_config(page_title="Image Retrieval Program", page_icon=":shark:", layout="wide", initial_sidebar_state='auto')

# # hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
# hide_streamlit_style = """
# 	<style>
#   #MainMenu {visibility: hidden;}
# 	footer {visibility: hidden;}
#   </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML
lottie_hello =  load_lottiefile(os.path.join(PARENT_DIRNAME, "lottiefiles/coding.json"))
loading = load_lottiefile(os.path.join(PARENT_DIRNAME, "lottiefiles/loading.json"))
st.lottie(lottie_hello, speed=1, width=200, height=200, key="initial")
st.title("Image Retrieval Program")
st.info("This program allows you to upload an image or take a picture using your camera and retrieve similar images from the database using different models.")

st.markdown("""
    <style>
    .st-emotion-cache-184fwp5 p, .st-emotion-cache-184fwp5 ol, .st-emotion-cache-184fwp5 ul, .st-emotion-cache-184fwp5 dl, .st-emotion-cache-184fwp5 li {
    font-size: x-large;
    font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

img_path = ""

with st.expander(label="Input Image", expanded=False):
    col1, col2 = st.columns([1, 3])
    method = 'Take a picture'

    with col1:
        method = st.radio('Select Input method',options=['Take a picture','Upload a picture'])
    
    with col2:

        if method == 'Take a picture':
            enable = st.checkbox('Enable Camera')
            img = st.camera_input('Take a picture',disabled= not enable)
        else:
            img = st.file_uploader('Upload a picture')

        if img is not None:
            img_path = save_image(img) #Save ori image
            img_path = process_image(img_path)
            if img_path is None:
                st.warning("Please take another image")
            else:
                st.success("Face cropped successfully")

        if st.button('Review Image (Cropped)'):
            if img is not None:
                img = cv2.imread(img_path,cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img,output_format='PNG')
                file_size = os.path.getsize(img_path)
                st.write(f"Size: {file_size} bytes")
                st.write("Image dimensions:", img.shape)
            else:
                st.warning("Please provide an image first")
if img_path:   
    img_path = os.path.join(PARENT_DIRNAME, img_path)

sys.path.append(os.path.join(PARENT_DIRNAME, "fine_tuning/"))
from fine_tuning.retrieval_models import RetrievalModel
from fine_tuning.img_retrieval_test import run_evaluation_pipeline_with_attributes
from fine_tuning.query_face_img import query_and_plot_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with st.expander("Select and Run model"):
    col1, col2 = st.columns([1, 3])
    with col1:
        model_name = st.selectbox('Select Model',options=['MobileNet_V2','ResNet50'])
        model_name = model_name.lower()
        
    if st.button('Run Model'):
        animation_placeholder = st.empty()

        with animation_placeholder:
            # st.write("Running model ...")
            st.lottie(loading, speed=1, width=400, height=400, key="loading")

        gallery_image_paths, distances = query_and_plot_images(
            query_image_path=img_path,
            model=model_name,
            top_k=9
        )
        animation_placeholder.empty()
        st.success("Model run successfully")
    


with st.expander("Output"):
    col1, col2 = st.columns([1,3])
    with col1:
        for i in range(20):
            st.write("\n")
        # Use HTML and CSS for styling
        caption = "Original Image"
        caption_html = """
        <div style="text-align: center; font-size: 20px; color: blue; font-weight: bold;">
            {caption}
        </div>
        """.format(caption=caption)

        st.markdown(caption_html, unsafe_allow_html=True)
        st.image(img_path)


    with col2:
        cols = cycle(st.columns(3)) 
        if gallery_image_paths is not None:
            for idx, filteredImage in enumerate(gallery_image_paths):
                caption = f'Rank: {idx}; Distance: {distances[idx]}'
                next(cols).image(filteredImage,caption=caption)
        