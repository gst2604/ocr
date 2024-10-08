import streamlit as st
import easyocr
import cv2
import numpy as np
from deep_translator import GoogleTranslator
from PIL import Image
import io
import re

reader = easyocr.Reader(['en', 'hi'], gpu=False)


st.title("Image Text Extraction and Translation")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    result = reader.readtext(img_cv)

    for detection in result:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        text = detection[1]
        img_cv = cv2.rectangle(img_cv, top_left, bottom_right, (0, 255, 0), 5)
        img_cv = cv2.putText(img_cv, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    extracted_text = ' '.join([detection[1] for detection in result])
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    def detect_language(text):
        if re.search(r'[\u0900-\u097F]', text): 
            return 'hi'
        return 'en'

    translated_text = ""
    for sentence in re.split(r'(?<=[.!?]) +', extracted_text): 
        language = detect_language(sentence)
        if language == 'hi':
            translated_text += GoogleTranslator(source='hi', target='en').translate(sentence) + " "
        else:
            translated_text += GoogleTranslator(source='en', target='hi').translate(sentence) + " "

    st.subheader("Translated Text (Mixed):")
    st.write(translated_text)

    _, processed_image_buffer = cv2.imencode('.png', img_cv)
    processed_image = Image.open(io.BytesIO(processed_image_buffer))

    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.image(processed_image, caption='Processed Image with Text Detection', use_column_width=True)

    search_term = st.text_input("Enter search term:")
    if search_term:
        highlighted_extracted_text = extracted_text.replace(search_term, f"<mark>{search_term}</mark>")
        highlighted_translated_text = translated_text.replace(search_term, f"<mark>{search_term}</mark>")

        st.subheader("Highlighted Extracted Text:")
        st.markdown(highlighted_extracted_text, unsafe_allow_html=True)

        st.subheader("Highlighted Translated Text:")
        st.markdown(highlighted_translated_text, unsafe_allow_html=True)

if st.button('Clear Uploads'):
    st.success("Uploads cleared!")
