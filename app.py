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

# Pełna lista etykiet klas
class_labels = ["Acanthoscurria", "Amazonius germani", "Aphonopelma seemanni", "Augcephalus", "Avicularia avicularia", "Avicularia juruensis", "Avicularia minatrix", "Avicularia purpurea", 
                "Birupes simoroxigorum", "Brachypelma albiceps", "Brachypelma auratum", "Brachypelma baumgarteni", "Brachypelma boehmei", "Brachypelma emilia", "Brachypelma hamorii or smithi", 
                "Brachypelma klaasi", "Bumba horrida or tapajos", "Caribena laeta", "Caribena versicolor", "Ceratogyrus brachycephalus", "Ceratogyrus darlingi", "Ceratogyrus marshalli", 
                "Ceratogyrus meridionalis", "Ceratogyrus sanderi", "Chilobrachys dyscolus", "Chilobrachys fimbriatus", "Chilobrachys huahini", "Chilobrachys natanicharum", "Chromatopelma cyaneopubescens", 
                "Cilantica devamatha", "Citharacanthus cyaneus", "Cyriocosmus aueri or bertae", "Cyriocosmus bicolor", "Cyriocosmus elegans", "Cyriocosmus leetzi", "Cyriocosmus perezmilesi", "Cyriocosmus ritae", 
                "Cyriopagopus (albostriatus, longipes, minax, paganus or vonwrithi", "Cyriopagopus hainanus", "Cyriopagopus lividus", "Cyriopagopus schmidti", "Davus", "Dolichothele diamantinensis", 
                "Encyocratella olivacea", "Ephebopus cyanognathus", "Ephebopus murinus", "Eucratoscelus pachypus", "Grammostola iheringi or actaeon", "Grammostola pulchra", "Grammostola pulchripes", 
                "Grammostola rosea", "Hapalopus", "Haplocosmia himalayana", "Harpactira cafreriana", "Harpactira pulchripes", "Heteroscodra maculata", "Heterothele gabonensis", "Holothele longipes", 
                "Homoeomma", "Hysterocrates", "Idiothele mira", "Kochiana brunnipes", "Lampropelma nigerrimum or Phormingochilus arboricola", "Lasiocyano sazimai", "Lasiodora", "Megaphobema robustum", 
                "Monocentropus balfouri", "Neoholothele incei", "Nhandu coloratovillosus", "Nhandu tripepii", "Omothymus schioedtei", "Omothymus violaceopes", "Ornithoctonus aureotibialis", 
                "Pamphobeteus antinous", "Pamphobeteus ultramarinus", "Pelinobus muticus", "Phormictopus auratus", "Phormingochilus everetti", "Poecilotheria", "Poecilotheria formosa", 
                "Poecilotheria metallica", "Poecilotheria ornata", "Poecilotheria rufilata", "Poecilotheria subfusca", "Psalmopoeus cambridgei", "Psalmopoeus irminia", "Psalmopoeus pulcher", 
                "Psalmopoeus reduncus", "Psalmopoeus victori", "Pterinochilus lugardi", "Pterinochilus murinus", "Selenobrachys philippinus", "Stromatopelma calceatum", "Tapinauchenius plumipes", "Theraphosa", 
                "Thrixopelma ockerti", "Tliltocatl albopilosus", "Tliltocatl vagans or kahlenbergi", "Typhochlaena seladonia", "Vitalius chromatus", "Xenesthis immanis"]

def download_model():
    """Pobiera model, jeśli jeszcze go nie ma."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with st.spinner('📥 Pobieranie modelu...'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=True)

@st.cache_resource
def load_trained_model():
    """Ładuje model Keras z dysku."""
    download_model()
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
         /* Górny pasek */
         .top-bar {{
             position: fixed;
             top: 0; left: 0; right: 0;
             height: 3.5rem;
             background-color: rgba(255,255,255,0.85);
             display: flex;
             justify-content: flex-end;
             align-items: center;
             padding: 0 1rem;
             z-index: 9999;
             gap: 1rem;
         }}
         .hamburger {{
             font-size: 1.5rem;
             cursor: pointer;
             user-select: none;
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

    # Tu przechowujemy stan rozwinięcia menu
    if "menu_open" not in st.session_state:
        st.session_state.menu_open = False

    # Pasek u góry - język + hamburger menu
    st.markdown(
        """
        <div class="top-bar">
            <div id="lang-select">
                <!-- Język będzie w Streamlit -->
            </div>
            <div id="hamburger" class="hamburger">&#9776;</div>
        </div>
        <script>
        const hamburger = window.parent.document.querySelector('.hamburger');
        hamburger.addEventListener('click', () => {
            const menu = window.parent.document.querySelector('#menu-container');
            if (menu.style.display === 'block') {
                menu.style.display = 'none';
            } else {
                menu.style.display = 'block';
            }
        });
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Ukryte menu - wyświetla się po kliknięciu hamburgera
    menu_container = st.empty()

    # Ponieważ JavaScript nie działa idealnie w streamlit, użyjemy tutaj uproszczonego rozwiązania:
    # pokażmy menu w sidebarowym stylu, ale jako "popup" poniżej paska.
    # Używamy checkboxa jako "toggle"
    with st.container():
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            menu_toggle = st.checkbox("", key="menu_toggle", label_visibility="collapsed")
        with col2:
            lang = st.selectbox("Language / Język", ["English", "Polski"], key="lang_top")

    if menu_toggle:
        page = st.radio(
            "Menu" if lang == "English" else "Menu",
            options=[
                "Prediction" if lang == "English" else "Predykcja",
                "Species List" if lang == "English" else "Lista gatunków",
                "Usage" if lang == "English" else "Instrukcja",
            ],
            key="page_radio",
        )
    else:
        # Domyślnie strona
        page = st.session_state.get("page_radio", "Prediction" if lang == "English" else "Predykcja")

    # Strona właściwa
    if page == ("Prediction" if lang == "English" else "Predykcja"):
        st.title("🕷️ Theraphosidae Species Classifier" if lang == "English" else "🕷️ Klasyfikator gatunków Theraphosidae")

        uploaded_file = st.file_uploader(
            "Upload an image (top view of full spider)" if lang == "English" else "Prześlij zdjęcie (cały pająk, widok z góry)",
            type=["jpg", "jpeg", "png"],
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
            st.markdown(
                """
            - Upload a photo of the full spider, taken from above.
            - The app recognizes only the species listed in the Species List tab.
            - If the species is not in the list but its genus is, the app will likely identify the genus correctly but assign the species to one from the list.
            """
            )
        else:
            st.markdown(
                """
            - Prześlij zdjęcie całego pająka, zrobione z góry.
            - Aplikacja rozpoznaje tylko gatunki wypisane na liście gatunków.
            - W przypadku gatunków nieobecnych na liście, ale obecnych ich rodzajów, aplikacja najprawdopodobniej prawidłowo rozpozna rodzaj, ale przypisze gatunek do jednego z dostępnych na liście.
            """
            )


if __name__ ==

