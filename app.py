import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown

# Ścieżki i link
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
MODEL_URL = "https://drive.google.com/uc?id=1fLsy6SAk-cGi5c06XVegJPjxfb0Bclxc"  # link gdrive

# Lista etykiet klas
class_labels = [  # przycięta dla czytelności – pełna w Twoim oryginale
    "Acanthoscurria", "Amazonius germani", "Aphonopelma seemanni", "Augcephalus",
    # ... cała lista ...
    "Vitalius chromatus", "Xenesthis immanis"
]

def download_model():
    """Pobiera model jeśli nie istnieje."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        with st.spinner('📥 Pobieranie modelu...'):
            os.makedirs(MODEL_DIR, exist_ok=True)
            gdown.download(MODEL_URL, MODEL_PATH, quiet=True)

@st.cache_resource
def load_trained_model():
    """Ładuje model Keras z dysku."""
    download_model()
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        st.error("❌ Nie udało się pobrać modelu.")
        st.stop()
    return load_model(MODEL_PATH)

model = load_trained_model()

def set_bg_hack_url():
    """Ustawia tło z obrazem."""
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
    """Generuje link do tarantupedia.com"""
    parts = name.lower().split()
    if len(parts) == 1:
        return f"https://www.tarantupedia.com/theraphosinae/{parts[0]}"
    else:
        genus = parts[0]
        species = '-'.join(parts)
        return f"https://www.tarantupedia.com/theraphosinae/{genus}/{species}"

def main():
    set_bg_hack_url()

    lang = st.sidebar.selectbox("Language / Język", ["English", "Polski"])

    # Poprawione przekazanie opcji do radio
    page = st.sidebar.radio(
        "Menu" if lang == "English" else "Menu",
        options=[
            "Prediction" if lang == "English" else "Predykcja",
            "Species List" if lang == "English" else "Lista gatunków",
            "Usage" if lang == "English" else "Instrukcja"
        ]
    )

    if page == ("Prediction" if lang == "English" else "Predykcja"):
        st.title("🕷️ Theraphosidae Species Classifier" if lang == "English" else "🕷️ Klasyfikator gatunków Theraphosidae")

        uploaded_file = st.file_uploader(
            "Upload an image (top view of full spider)" if lang == "English" else "Prześlij zdjęcie (cały pająk, widok z góry)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_label = class_labels[predicted_index]

            url = tarantupedia_link(predicted_label)
            info_text = "Click to learn more" if lang == "English" else "Kliknij, aby dowiedzieć się więcej"

            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"### ✅ Prediction: [{predicted_label}]({url})")
            st.markdown(f"*{info_text}*")

    elif page == ("Species List" if lang == "English" else "Lista gatunków"):
        st.title("Recognized Species" if lang == "English" else "Rozpoznawane gatunki")
        st.write(
            "The model recognizes the following species and genera (mainly from the pet trade):"
            if lang == "English"
            else "Model rozpoznaje następujące gatunki i rodzaje (głównie popularne w handlu):"
        )
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
