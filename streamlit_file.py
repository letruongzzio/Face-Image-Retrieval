import streamlit as st
import io
import time
import PIL.Image
import sys
import cv2
sys.path.append('/home/tiamo/Documents/code/Digital Image Processing/image-processing-project/lib')
from processing import crop_face


def save_image(img):
    img_folder = '/home/tiamo/Documents/code/Digital Image Processing/image-processing-project/img'
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    img_path = f"{img_folder}/{timestamp}.png"

    with open(img_path, 'wb') as f:
        f.write(img.getvalue())
        st.success(f"Image saved to {img_path}")
    return img_path



def process_image(image_path):
    image = cv2.imread(image_path)
    state, cropped_image = crop_face(image, 1.12, 12)
    if state == "one":
        cv2.imwrite(image_path, cropped_image)
        return img_path
    else:
        return None


# Design the layout of the Streamlit app
st.set_page_config(page_title="Image Retrieval Program", page_icon=":shark:", layout="wide", initial_sidebar_state='auto')

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

# page_bg_img = """
# <style>
# [data-testid="stAppViewContainer"]{
#   background-image: url("https://plus.unsplash.com/premium_photo-1671995576541-a63078fa3063?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
#   background-size: cover;
# }
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# <style>

# """
# st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Image Retrieval Program")
st.info("This program allows you to upload an image or take a picture using your camera and retrieve similar images from the database using different models.")

st.subheader("Input Image")
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
        # if img_path is not None:
        #     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     st.image(img, output_format='PNG')
        # else:
        #     st.warning("Please provide an image first")
        # st.write(f"Size: {img.size} bytes")
        # img_file = io.BytesIO(cv2.imencode('.png', img)[1])
        # image = PIL.Image.open(img_file)
        # st.write("Image dimensions:", image.size)
        # st.write("Image mode:", image.mode)
        # if method == 'Upload a picture':
        #     st.write(f"Filename: {img_path}")
        #     st.image(img_path, caption='Uploaded Image', width=200, use_column_width=True)
    else:
        st.warning("Please provide an image first")


st.subheader("Selecting Model")
col1, col2 = st.columns([1, 3])
with col1:
  model = st.selectbox('Select Model',options=['MobileNet_V2','ResNet50'])
  model = model.lower()

st.subheader("Output")

