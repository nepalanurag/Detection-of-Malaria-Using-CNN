import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np
model = load_model("malaria_cnn.h5")
st.title("Malarial Parasite Detection")
st.text("Detect if you have malaria or not")
st.markdown("Upload an image of the blood smear")
img=st.file_uploader("")
submit = st.button('Predict')
st.sidebar.title('Developers Contact')
st.sidebar.markdown('[![Avishek Rijal]'
                        '(https://img.shields.io/badge/Author-Avishek--Rijal-lightgrey)]'
                        '(https://www.linkedin.com/in/avishek-rijal-0430801b1/)')
                        
if submit:


    if img is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (32,32))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,32,32,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        ans = np.argmax(Y_pred)
        if(ans == 1):
            result = "not infected"
        else:
            result = "Infected"

        st.title(str("The model predicts the blood smear is "+result ))