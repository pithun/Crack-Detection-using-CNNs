# Imports
import cv2
import streamlit as st


# Data Manipulation
import numpy as np
from Functions import Pipeline

st.title(""":blue[Pi-segment.AI]""")
st.write('Welcome to Pi-Segment, a product of Pithun-Corp.AI who focuses on innovative application of Artificial'
         ' Intelligence to the domain of Civil Engineering to enhance all areas of it from Structural '
         'optimization to health monitoring to population forecast to aid optimal structure planning.')

st.write('Pi-Segment performs image and video segmentation of cracks on videos and images using preprocessing'
         ' techniques.')

# Creating the option to upload image or video
file_type = ['Image', 'Video']
option_file_type = st.selectbox('Upload Video or Image?', options=file_type)
file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.write('Segmenting in a sec...')
    seg_img = Pipeline(img)
    st.image(seg_img)


