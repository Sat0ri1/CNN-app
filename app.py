import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests

# Model download function
def download_model():
    url = 'https://drive.google.com/uc?export=download&id=1fLsy6SAk-cGi5c06XVegJPjxfb0Bclxc'
    os.makedirs('model', exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open('model/model.h5', 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# Load model once, download if not exists
@st.cache_resource
def load_trained_model():
    if not os.path.exists('model/model.h5'):
        with st.spinner('Downloading model, please wait...'):
            download_model()
    return load_model('model/model.h5')

model = load_trained_model()

# Class labels (skrót - wrzuć pełną listę do pliku albo tu)
class_labels = [
    "Acanthoscurria", "Amazonius germani", "Aphonopelma seemanni",
    "Augcephalus", "Avicularia avicularia", "Avicularia juruensis",
    "Poecilotheria", "Poecilotheria metallica"
    # ... dodaj resztę
]

def set_bg_hack_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://i.imgur.com/DBGp1Yv.png");
             background-size: cover;
             background-position: top right 18vw;
             background-repeat: no-repeat;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def tarantupedia_link(name):
    parts = name.lower().split()
    if len(parts) == 1:
        # genus only
        return f"https://www.tarantupedia.com/theraphosinae/{parts[0]}"
    else:
        # genus + species
        genus = parts[0]
        species = '-'.join(parts)
        return f"https://www.tarantupedia.com/theraphosinae/{genus}/{species}"

def main():
    set_bg_hack_url()
    
    # Language selection
    lang = st.sidebar.selectbox("Language / Język", ["English", "Polski"])

    # Sidebar navigation
    page = st.sidebar.radio(
        "Menu" if lang == "English" else "Menu",
        ("Prediction" if lang == "English" else "Predykcja"),
        ("Species List" if lang == "English" else "Lista gatunków"),
        ("Usage" if lang == "English" else "Instrukcja")
    )
    
    if page == ("Prediction" if lang == "English" else "Predykcja"):
        st.title("Theraphosidae Species Classifier" if lang == "English" else "Klasyfikator gatunków Theraphosidae")

        uploaded_file = st.file_uploader("Upload an image (top view of full spider)" if lang == "English" else "Prześlij zdjęcie (cały pająk, widok z góry)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            img = image.load_img(uploaded_file, target_size=(224,224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_label = class_labels[predicted_index]

            url = tarantupedia_link(predicted_label)
            info_text = "Click to learn more" if lang == "English" else "Kliknij, aby dowiedzieć się więcej"

            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"### Prediction: [{predicted_label}]({url})")
            st.markdown(f"*{info_text}*")
    
    elif page == ("Species List" if lang == "English" else "Lista gatunków"):
        st.title("Recognized Species" if lang == "English" else "Lista rozpoznawanych gatunków")
        st.write("The model recognizes the following species and genus (mostly popular in pet trade):" if lang == "English" else "Model rozpoznaje następujące gatunki i rodzaje (głównie popularne w handlu):")
        st.write(", ".join(class_labels))
    
    elif page == ("Usage" if lang == "English" else "Instrukcja"):
        st.title("Usage Instructions" if lang == "English" else "Instrukcja użycia")
        if lang == "English":
            st.markdown("""
            - Upload a photo of the full spider, taken from above.
            - The app recognizes only the species listed in the Species List tab.
            - If the species is not in the list but its genus is, the app will likely identify the genus correctly but assign the species to one from the list.
            """)
        else:
            st.markdown("""
            - Prześlij zdjęcie całego pająka, zrobione z góry.
            - Aplikacja rozpoznaje tylko gatunki wypisane na liście gatunków.
            - W przypadku gatunków nieobecnych na liście, ale obecnych ich rodzajów, aplikacja najprawdopodobniej prawidłowo rozpozna rodzaj, ale przypisze gatunek do jednego z dostępnych na liście.
            """)

if __name__ == "__main__":
    main()

