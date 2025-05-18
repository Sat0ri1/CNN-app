import streamlit as st
import streamlit.components.v1 as components
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown

# Twoje dane (tu wstaw pełną listę gatunków)
class_labels = [
    "Aphonopelma chalcodes",
    "Brachypelma smithi",
    "Grammostola rosea",
    # ... pełna lista gatunków ...
]

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
MODEL_URL = "https://drive.google.com/uc?id=1fLsy6SAk-cGi5c06XVegJPjxfb0Bclxc"

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

def get_query_params():
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["prediction"])[0]
    lang = query_params.get("lang", ["en"])[0]
    return page, lang

def render_top_bar(page, lang):
    html = f"""
    <style>
    .topnav {{
        position: fixed;
        top: 0;
        right: 0;
        width: 100%;
        height: 50px;
        background-color: #f0f2f6;
        border-bottom: 1px solid #ccc;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 0 20px;
        z-index: 9999;
        font-family: sans-serif;
    }}
    .topnav select {{
        margin-right: 10px;
        padding: 4px;
        font-size: 14px;
    }}
    .dropdown {{
        position: relative;
        display: inline-block;
    }}
    .dropbtn {{
        font-size: 24px;
        background: none;
        border: none;
        cursor: pointer;
        user-select: none;
    }}
    .dropdown-content {{
        display: none;
        position: absolute;
        right: 0;
        background-color: #f9f9f9;
        min-width: 160px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        z-index: 10000;
    }}
    .dropdown-content a {{
        color: black;
        padding: 12px 16px;
        text-decoration: none;
        display: block;
    }}
    .dropdown-content a:hover {{
        background-color: #ddd;
    }}
    .dropdown:hover .dropdown-content {{
        display: block;
    }}
    </style>
    <div class="topnav">
        <select onchange="window.location.search='lang='+this.value+'&page={page}'">
            <option value="en" {"selected" if lang == "en" else ""}>English</option>
            <option value="pl" {"selected" if lang == "pl" else ""}>Polski</option>
        </select>
        <div class="dropdown">
            <button class="dropbtn">&#9776;</button>
            <div class="dropdown-content">
                <a href="?lang={lang}&page=prediction">{'Prediction' if lang == 'en' else 'Predykcja'}</a>
                <a href="?lang={lang}&page=species">{'Species List' if lang == 'en' else 'Lista gatunków'}</a>
                <a href="?lang={lang}&page=usage">{'Usage' if lang == 'en' else 'Instrukcja'}</a>
            </div>
        </div>
    </div>
    <div style="height: 60px;"></div>
    """
    components.html(html, height=70)

def main():
    page, lang = get_query_params()
    render_top_bar(page, lang)

    if page == "prediction":
        st.title("🕷️ Theraphosidae Species Classifier" if lang == "en" else "🕷️ Klasyfikator gatunków Theraphosidae")

        uploaded_file = st.file_uploader(
            "Upload an image (top view of full spider)" if lang == "en" else "Prześlij zdjęcie (cały pająk, widok z góry)",
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
            info_text = "Click to learn more" if lang == "en" else "Kliknij, aby dowiedzieć się więcej"

            st.image(uploaded_file, caption="Uploaded Image")
            st.markdown(f"### ✅ Prediction: [{predicted_label}]({url})")
            st.markdown(f"*{info_text}*")

    elif page == "species":
        st.title("Recognized Species" if lang == "en" else "Rozpoznawane gatunki")
        st.write(
            "The model recognizes the following species and genera (mainly from the pet trade):"
            if lang == "en"
            else "Model rozpoznaje następujące gatunki i rodzaje (głównie popularne w handlu):"
        )
        st.write(", ".join(class_labels))

    elif page == "usage":
        st.title("Usage Instructions" if lang == "en" else "Instrukcja użycia")
        st.markdown("""
        - Upload a photo of the full spider, taken from above.
        - The app recognizes only the species listed in the Species List tab.
        - If the species is not in the list but its genus is, the app will likely identify the genus correctly but assign the species to one from the list.
        """ if lang == "en" else """
        - Prześlij zdjęcie całego pająka, zrobione z góry.
        - Aplikacja rozpoznaje tylko gatunki wypisane na liście gatunków.
        - W przypadku gatunków nieobecnych na liście, ale obecnych ich rodzajów, aplikacja najprawdopodobniej prawidłowo rozpozna rodzaj, ale przypisze gatunek do jednego z dostępnych na liście.
        """)

if __name__ == "__main__":
    main()

