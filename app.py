import streamlit as st
import easyocr
import cv2
import numpy as np
from deep_translator import GoogleTranslator
from PIL import Image
import io
from gtts import gTTS
import base64
import fitz  

def split_text(text, max_length=5000):
    """
    Splits text into chunks of a specified maximum length.
    """
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def get_ocr_languages(target_language):
    return [target_language, 'en']


st.title("Image/PDF Text Extraction, Translation, and Audio Output")

language_selection = st.selectbox(
    "Choose the target language for translation:",
    ["English", "Hindi", "Spanish", "French", "German", "Italian"]
)

language_mapping = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it"
}

target_language = language_mapping[language_selection]
ocr_languages = get_ocr_languages(target_language)

reader = easyocr.Reader(ocr_languages, gpu=False)

uploaded_file = st.file_uploader("Choose a file (image or PDF)...", type=['jpg', 'jpeg', 'png', 'pdf'])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        st.write("Uploaded file is a PDF.")

        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        extracted_text = ""

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            extracted_text += page_text + "\n"

            pix = page.get_pixmap()
            pdf_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_cv = cv2.cvtColor(np.array(pdf_image), cv2.COLOR_RGB2BGR)
            result = reader.readtext(img_cv)
            for detection in result:
                text = detection[1]
                extracted_text += text + " "

        pdf_document.close()

    elif file_type in ["image/jpeg", "image/png"]:
        st.write("Uploaded file is an image.")

        image = Image.open(uploaded_file)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresholded_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blurred_img = cv2.GaussianBlur(thresholded_img, (5, 5), 0)

        result = reader.readtext(blurred_img)

        extracted_text = ""
        for detection in result:
            top_left = tuple(map(int, detection[0][0]))
            bottom_right = tuple(map(int, detection[0][2]))
            text = detection[1]
            extracted_text += text + " "
            img_cv = cv2.rectangle(img_cv, top_left, bottom_right, (0, 255, 0), 5)
            img_cv = cv2.putText(img_cv, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    else:
        st.write("Unsupported file type.")
        extracted_text = ""

    st.subheader("Extracted Text:")
    st.write(extracted_text.strip())

    if not extracted_text.strip() or extracted_text.strip().isnumeric():
        st.warning("The extracted text seems invalid or unreadable. Please check the quality of the uploaded file.")

    if extracted_text.strip():
    
        text_chunks = split_text(extracted_text.strip())

        translated_text = ""
        for i, chunk in enumerate(text_chunks):
            st.write(f"Processing chunk {i + 1} with {len(chunk)} characters...")
            try:
                translation = GoogleTranslator(source='auto', target=target_language).translate(chunk)
                translated_text += translation + " "
            except Exception as e:
                st.error(f"Error translating chunk {i + 1}: {e}")
    else:
        translated_text = ""

    st.subheader(f"Translated Extracted Text ({language_selection}):")
    st.write(translated_text.strip())

    try:
        tts = gTTS(translated_text.strip(), lang=language_mapping[language_selection])
        tts.save("translated_audio.mp3")
        audio_file = open("translated_audio.mp3", "rb").read()
        audio_base64 = base64.b64encode(audio_file).decode()

        audio_html = f'<audio autoplay="true" controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating audio: {e}")

    if file_type in ["image/jpeg", "image/png"]:
        _, processed_image_buffer = cv2.imencode('.png', img_cv)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.image(io.BytesIO(processed_image_buffer), caption='Processed Image with Text Detection', use_column_width=True)

    search_term = st.text_input("Enter search term:")
    if search_term:
        highlighted_extracted_text = extracted_text.replace(search_term, f"<mark>{search_term}</mark>")
        highlighted_translated_text = translated_text.replace(search_term, f"<mark>{search_term}</mark>")

        st.subheader("Highlighted Extracted Text:")
        st.markdown(highlighted_extracted_text, unsafe_allow_html=True)

        st.subheader("Highlighted Translated Text:")
        st.markdown(highlighted_translated_text, unsafe_allow_html=True)

    if st.button('Clear Uploads'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()  
