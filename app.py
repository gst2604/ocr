import streamlit as st
import easyocr
import cv2
import numpy as np
import os
from deep_translator import GoogleTranslator
from PIL import Image

# Create EasyOCR reader
reader = easyocr.Reader(['en', 'hi'], gpu=False)

# Upload folder setup (optional, as Streamlit handles file uploads in-memory)
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Streamlit app
st.title("Image Text Extraction and Translation")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Save uploaded file
    image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read and process the image
    result = reader.readtext(image_path)
    img = cv2.imread(image_path)

    for detection in result:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        text = detection[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)
        img = cv2.putText(img, text, top_left, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Save processed image
    processed_image_path = os.path.join(UPLOAD_FOLDER, 'processed_' + uploaded_file.name)
    cv2.imwrite(processed_image_path, img)

    # Extracted text
    extracted_text = ' '.join([detection[1] for detection in result])
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Try to translate the extracted text to Hindi using deep-translator
    try:
        translated_text = GoogleTranslator(source='en', target='hi').translate(extracted_text)
    except Exception as e:
        translated_text = "Translation failed due to: " + str(e)

    st.subheader("Translated Text (Hindi):")
    st.write(translated_text)

    # Display the images
    st.image(Image.open(image_path), caption='Uploaded Image', use_column_width=True)
    st.image(Image.open(processed_image_path), caption='Processed Image with Text Detection', use_column_width=True)

    # Search functionality
    search_term = st.text_input("Enter search term:")
    if search_term:
        highlighted_extracted_text = extracted_text.replace(search_term, f"<mark>{search_term}</mark>")
        highlighted_translated_text = translated_text.replace(search_term, f"<mark>{search_term}</mark>")

        st.subheader("Highlighted Extracted Text:")
        st.markdown(highlighted_extracted_text, unsafe_allow_html=True)

        st.subheader("Highlighted Translated Text:")
        st.markdown(highlighted_translated_text, unsafe_allow_html=True)

# Option to clear uploads (optional)
if st.button('Clear Uploads'):
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            os.remove(file_path)
    st.success("Uploads cleared!")