# Contents of ~/my_app/pages/page_3.py
import streamlit as st
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import tensorflow as tf
import numpy as np
model = load_model("my_model.h5")
st.markdown(
    f'<style>.reportview-container .main .block-container {{max-width: 800px;}}</style>',
    unsafe_allow_html=True,
)

# Set page header
st.markdown("<h1 style='text-align: center;'>Malarial Parasite Detection</h1>", unsafe_allow_html=True)

# Set form header
st.markdown("<h2 style='text-align: center;'>Please enter your details</h2>", unsafe_allow_html=True)

st.markdown("###")

# Set form columns
col1, col2 = st.columns(2)

# Set form fields
with col1:
    st.subheader("Name")
    name = st.text_input("", value="Your Full Name")

    st.subheader("Age")
    age = st.text_input("", value="Age")

with col2:
    st.subheader("Location")
    location = st.selectbox("", ["", "Bangalore"])

    st.subheader("Phone Number")
    no = st.text_input("", value="Phone No.")

# Add image upload
st.markdown("<h2 style='text-align: center;'>Upload an image of the blood smear</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Please upload a high-quality image of the blood smear for accurate diagnosis.</p>", unsafe_allow_html=True)
img = st.file_uploader("", type=['jpg', 'jpeg', 'png'])



##img=st.file_uploader("*Here*")




submit = st.button('Predict')                        





if submit:


    if img is not None:


        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)


        file_img = opencv_image
        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (50,50))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,50,50,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        print(Y_pred)
        ans = np.argmax(Y_pred)
        if(ans == 1):
            result = "not infected"
            st.title(str("The model predicts the blood smear is "+result ))
        else:
            
            result = "Infected"
            st.title(str("The model predicts the blood smear is "+result ))
            st.markdown("Process of detecting malarial parasite in blood")

            def change_contrast(image,alpha = 1.5,beta=25):
                adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                return adjusted
            def image_cleanup(image):
                blurred = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
                thresh = cv2.threshold(blurred, 175, 250, cv2.THRESH_BINARY)[1]
                return thresh
            def remove_noisy_regions(image):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
                element = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
                mask = cv2.erode(gray, element, iterations = 50)
                mask = cv2.dilate(mask, element, iterations = 50)
                element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                mask = cv2.erode(gray, element, iterations = 1)
                mask = cv2.dilate(mask, element, iterations = 1)
                mask = cv2.erode(mask, element)
                gray = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
                return gray
            def find_contours_and_centers(img_input):
                contours_raw, hierarchy = cv2.findContours(img_input, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                contours = [i for i in contours_raw]
                contour_centers = []
                
                for idx, c in enumerate(contours):
                    M = cv2.moments(c)
                    try:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    except:
                        cX = int(M["m10"] / (M["m00"] + 0.0001))
                        cY = int(M["m01"] / (M["m00"] + 0.0001))
                    samp_bounds = cv2.boundingRect(c)
                    contour_centers.append(((cX,cY), samp_bounds))
                contour_centers = sorted(contour_centers, key=lambda x: x[0])

                return (contours, contour_centers)    
            def find_contour_and_area(img_input):
                plt.axis('off')
                lll=img_input
                i1 = change_contrast(lll,1.5,10)
                i2 = image_cleanup(i1)
                img = remove_noisy_regions(i2)
                h,s,v = cv2.split(img)
                ab = cv2.subtract(h,s)
                fig,asx = plt.subplots(1,3)
                plt.yticks([])
                plt.xticks([])
                asx[0].imshow(h,cmap="gray")
                asx[1].imshow(s,cmap="gray")
                asx[2].imshow(ab,cmap="gray")
                np.vectorize(lambda ax:ax.axis('off'))(asx)
                st.pyplot(fig)
                fig.tight_layout()
                #plt.savefig("processing.png",dpi=300)
                conts, cents = find_contours_and_centers(s)
                # circles = [i for i in conts if np.logical_and((cv2.contourArea(i) > 650),(cv2.contourArea(i) < 4000))]
                print(len(cents))
                img = lll.copy()
                cv2.drawContours(img, conts, -1, (0,255,0), 2)
                # show_image(img)

                plt.axis('off')
                st.write("Outlining the parasite")
                fig,asx = plt.subplots(1,2)
                asx[0].imshow(lll)
                asx[1].imshow(img,cmap="gray")
                np.vectorize(lambda ax:ax.axis('off'))(asx)

                fig.tight_layout()
                st.pyplot(fig)
                #plt.savefig("contour_infected.png",dpi=300)
                if(len(cents)<2):
                    st.write("Fused with white blood cells")
                    ratio=0
                if(len(cents)==2):
                    st.write("Model predicts that there is one parasite in the blood sample")
                    ratio=cv2.contourArea(conts[1])/cv2.contourArea(conts[0])*100
                    st.write("The area of the blood sample is ",cv2.contourArea(conts[0]))
                    st.write("The aree of parasite is ",cv2.contourArea(conts[1]))
                elif (len(cents)>2):
                    area=0
                    for i in range(len(cents)):
                        if (i==0):continue
                        area+=cv2.contourArea(conts[i])
                    ratio=(area/cv2.contourArea(conts[0]))*100
                    st.write("Model predicts that there are more than one parasite in the blood sample",len(cents)-1)
                    st.write("The area of the blood sample is ",cv2.contourArea(conts[0]))
                    st.write("The area of parasite is ",area)
                return ratio
            
            ratio = (find_contour_and_area(file_img))
            
            if(ratio>=7.0):
                st.markdown("# Advanced Schizont Stage")
            elif(ratio>=2.5 and ratio<=7.0):
                st.markdown("# Trophozoite Stage")
            else:    
                st.markdown("# Initial Ring Stage")

        c = canvas.Canvas("reports.pdf",pagesize=(200,250),bottomup=0)
        # Logo Section
        # Setting th origin to (10,40)
        c.translate(10,40)
        # Inverting the scale for getting mirror Image of logo
        c.scale(1,-1)
        # Inserting Logo into the Canvas at required position
        c.drawImage("logo.jpeg",0,0,width=30,height=30)
        # Title Section
        # Again Inverting Scale For strings insertion
        c.scale(1,-1)
        # Again Setting the origin back to (0,0) of top-left
        c.translate(-10,-40)
        # Setting the font for Name title of company
        c.setFont("Helvetica-Bold",10)
        # Inserting the name of the company
        c.drawCentredString(125,20,"NITTE MEENAKSHI")
        # For under lining the title
        c.line(70,22,180,22)
        # Changing the font size for Specifying Address
        c.setFont("Helvetica-Bold",5)
        c.drawCentredString(125,30,"Malaria Detection APP")
        c.drawCentredString(125,35,"21CSP20")
        # Changing the font size for Specifying GST Number of firm
        c.setFont("Helvetica-Bold",6)
        c.drawCentredString(125,42,"GUIDE : Dr. Nalini N")
        # Line Seprating the page header from the body
        c.line(5,45,195,45)
        # Document Information
        # Changing the font for Document title
        c.setFont("Courier-Bold",8)
        c.drawCentredString(100,55,"Lab Reports")
        # This Block Consist of Costumer Details
        c.roundRect(15,63,170,60,10,stroke=1,fill=0)
        c.setFont("Times-Bold",5)
        c.drawCentredString(100,70,"Lab Report No. : XXXXXXXXXX")
        c.drawCentredString(100,80,"DATE           : 2023-04-05")
        c.drawCentredString(100,90,"Patient NAME  : "+name)
        c.drawCentredString(100,100,"PHONE No.     : "+no)
        c.drawCentredString(100,110,"Location      : "+location)
        c.drawCentredString(100,120,"Age           : "+age)

        # This Block Consist of Item Description
        
        if ans==1:
            c.drawCentredString(100,135,"RESULTS = SAFE // NOT INFECTED")
            c.drawImage("safe.png",90,160,width=30,height=30)
        else:
            c.drawCentredString(100,135,"RESULTS = Infected With MALARIA")
            c.drawImage("infected.png",90,160,width=30,height=30)
        
        # Declaration and Signature
        c.line(15,220,185,220)
        c.line(100,220,100,238)
        c.drawString(20,225,"We declare that above mentioned")
        c.drawString(20,230,"information is true.")
        c.drawString(20,235,"(Electronically generated REPORT)")
        c.drawRightString(169,230,"Authorised Signatory")
        # End the Page and Start with new
        c.showPage()
        # Saving the PDF
        c.save()
                            
    else:
        st.warning('No image uploaded', icon="⚠️")

        #ratio_ans = "The ratio of blood to parasite is " + str(ratio)
        #st.markdown(ratio_ans)