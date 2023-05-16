import tensorflow as tf
model = tf.keras.models.load_model('my_model.hdf5')

import streamlit as st
st.write("""
         # Skin Cancer Predictor
         """
         )
st.write("This is a simple image classification web app to predict whether you have Skin cancer")
file = st.file_uploader("Please upload of defected area", type=["jpg", "png"])


import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img=image
        img_resize = (cv2.resize(img, dsize=(244, 244),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if (prediction) > 0:
        st.write("Cancer")
    elif (prediction) <0:
        st.write("Not Cancer ")
   
    
    