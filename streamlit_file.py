import streamlit as st
import io
import PIL.Image

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

  if st.button('Review Image'):
    if img is not None:
      st.write("Image Details:")
      st.write(f"Format: {img.type}")
      st.write(f"Size: {img.size} bytes")
      img_file = io.BytesIO(img.getvalue())
      image = PIL.Image.open(img_file)
      st.write("Image dimensions:", image.size)
      st.write("Image mode:", image.mode)
      if method == 'Upload a picture':
        st.write(f"Filename: {img.name}")
      st.image(img, caption='Uploaded Image', width=200,use_container_width=True)
    else:
      st.warning("Please provide an image first")
st.subheader("Selecting Model")
col1, col2 = st.columns([1, 3])
with col1:
  model = st.selectbox('Select Model',options=['VGG16','ResNet50','InceptionV3'])

st.subheader("Output")