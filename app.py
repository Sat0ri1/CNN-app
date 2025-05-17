import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown

# Twoje dane (skrócone dla przejrzystości)
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
MODEL_URL = "https://drive.google.com/uc?id=1fLsy6SAk-cGi5c06XVegJPjxfb0Bclxc"

class_labels = [
    # tu wklej pełną listę gatunków
]

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with st.spinner('📥 Pobieranie modelu...'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=True)

@st.cache_resource
def load_trained_model():
    download_model()
    return load_model(MODEL_PATH)

model = load_trained_model()

def tarantupedia_link(name):
    parts = name.lower().split()
    if len(parts) == 1:
        return f"https://www.tarantupedia.com/theraphosinae/{parts[0]}"
    else:
        genus = parts[0]
        species = '-'.join(parts)
        return f"https://www.tarantupedia.com/theraphosinae/{genus}/{species}"

def main():
    # Pasek top fixed, poza kontenerem
    st.markdown(
        """
        <style>
        /* Ukrywa domyślny nagłówek streamlit */
        header {visibility: hidden;}
        
        /* Pasek na górze poza kontenerem */
        .top-bar-outside {
            position: fixed;
            top: 0;
            right: 0;
            left: 0;
            height: 50px;
            background-color: #f0f2f6;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: flex-end;
            align-items: center;
            padding: 0 20px;
            z-index: 10000;
            font-family: Arial, sans-serif;
            gap: 20px;
        }

        /* Trochę odstępu od góry dla kontenera streamlit, żeby nie przykrywał paska */
        .appview-container {
            padding-top: 60px !important;
        }

        /* Styl selectbox (w miarę możliwości) */
        div[data-baseweb="select"] > div {
            min-width: 150px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Wyświetlamy "pasek" jako HTML + widgety Streamlit w jednej linii na prawo
    # Niestety, streamlit nie pozwala na renderowanie widgetów w raw HTML, więc:
    # możemy zrobić pseudo-pasek na górze i w nim wyrenderować widgety

    # Do tego celu używamy kolumn, ale muszą być poza głównym kontenerem,
    # więc trick: najpierw st.empty() i potem tam wstawiamy widgety.

    # Stwórz placeholder dla paska (możemy wyrenderować w nim widgety)
    top_bar = st.container()
    with top_bar:
        # Ustawiamy pasek w html
        st.markdown(
            """
            <div class="top-bar-outside">
            """,
            unsafe_allow_html=True,
        )

        # Widgety język i menu muszą być Streamlitowe, więc renderujemy je na samym dole tej sekcji
        cols = st.columns([1,1])
        with cols[0]:
            lang = st.selectbox("", ["English", "Polski"], key="lang_outside", label_visibility="collapsed")
        with cols[1]:
            page = st.selectbox(
                "",
                options=[
                    "Prediction" if lang == "English" else "Predykcja",
                    "Species List" if lang == "English" else "Lista gatunków",
                    "Usage" if lang == "English" else "Instrukcja"
                ],
                key="page_outside",
                label_visibility="collapsed",
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Teraz reszta strony normalnie w kontenerze streamlit, ale z paddingiem od góry (z CSS wyżej)
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

            st.image(uploaded_file, caption="Uploaded Image")
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
