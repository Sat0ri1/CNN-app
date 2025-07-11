import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import gdown

# Ścieżki i link
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
MODEL_URL = "https://drive.google.com/uc?export=download&id=19bxCSLca5ygQxnxHt5lsksQclXZ_Ed-S"  # link gdrive

# Pełna lista etykiet klas
class_labels = ["Acanthoscurria geniculata", "Amazonius germani", "Aphonopelma seemanni", "Augcephalus", "Avicularia avicularia", "Avicularia juruensis", "Avicularia minatrix", "Avicularia purpurea", 
                "Birupes simoroxigorum", "Brachypelma albiceps", "Brachypelma auratum", "Brachypelma baumgarteni", "Brachypelma boehmei", "Brachypelma emilia", "Brachypelma hamorii or smithi", 
                "Brachypelma klaasi", "Bumba horrida or tapajos", "Caribena laeta", "Caribena versicolor", "Ceratogyrus brachycephalus", "Ceratogyrus darlingi", "Ceratogyrus marshalli", 
                "Ceratogyrus meridionalis", "Ceratogyrus sanderi", "Chilobrachys dyscolus", "Chilobrachys fimbriatus", "Chilobrachys huahini", "Chilobrachys natanicharum", "Chromatopelma cyaneopubescens", 
                "Cilantica devamatha", "Citharacanthus cyaneus", "Cyriocosmus aueri or bertae", "Cyriocosmus bicolor", "Cyriocosmus elegans", "Cyriocosmus leetzi", "Cyriocosmus perezmilesi", "Cyriocosmus ritae", 
                "Cyriopagopus albostriatus", "Cyriopagopus hainanus", "Cyriopagopus lividus", "Cyriopagopus schmidti", "Davus pentaloris", "Dolichothele diamantinensis", 
                "Encyocratella olivacea", "Ephebopus cyanognathus", "Ephebopus murinus", "Eucratoscelus pachypus", "Grammostola iheringi", "Grammostola pulchra", "Grammostola pulchripes", 
                "Grammostola rosea", "Hapalopus formosus", "Haplocosmia himalayana", "Harpactira cafreriana", "Harpactira pulchripes", "Heteroscodra maculata", "Heterothele gabonensis", "Holothele longipes", 
                "Homoeomma", "Hysterocrates gigas", "Idiothele mira", "Kochiana brunnipes", "Lampropelma nigerrimum or Phormingochilus arboricola", "Lasiocyano sazimai", "Lasiodora", "Megaphobema robustum", 
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
             background: url("https://i.imgur.com/27EU8Ta.png");
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def tarantupedia_link(name):
    """Generuje link do tarantupedia.com. 
    Jeśli w nazwie występuje 'or', przekierowuje tylko do rodzaju (genus).
    """
    name = name.strip()

    if ' or ' in name.lower():
        # Zawsze bierzemy tylko pierwszy wyraz jako genus
        genus = name.split()[0].lower()
        return f"https://www.tarantupedia.com/theraphosinae/{genus}"

    parts = name.lower().split()
    if len(parts) == 1:
        return f"https://www.tarantupedia.com/theraphosinae/{parts[0]}"
    elif len(parts) == 2:
        return f"https://www.tarantupedia.com/theraphosinae/{parts[0]}-{parts[1]}"
    else:
        return f"https://www.tarantupedia.com/theraphosinae/{parts[0]}"


def main():
    set_bg_hack_url()

    lang = st.sidebar.selectbox("Language / Język", ["English", "Polski"])

    page = st.sidebar.radio(
        "Menu" if lang == "English" else "Menu",
        options=[
            "Prediction" if lang == "English" else "Predykcja",
            "Species List" if lang == "English" else "Lista gatunków",
            "Usage" if lang == "English" else "Instrukcja",
            "Credits" if lang == "English" else "Podziękowania"
        ]
    )

    if page == ("Prediction" if lang == "English" else "Predykcja"):
        st.title("Theraphosidae Species Classifier" if lang == "English" else "Klasyfikator gatunków Theraphosidae")

        uploaded_file = st.file_uploader(
            "Upload an image (top view of full spider)" if lang == "English" else "Prześlij zdjęcie (cały pająk, widok z góry)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            img = image.load_img(uploaded_file, target_size=(299, 299))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
            img_array = preprocess_input(img_array)

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_label = class_labels[predicted_index]

            url = tarantupedia_link(predicted_label)
            info_text = "Click to learn more" if lang == "English" else "Kliknij, aby dowiedzieć się więcej"

            st.image(uploaded_file, caption="Uploaded Image")
            st.markdown(f"### Prediction: [{predicted_label}]({url})")
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
            - App recognizes tarantulas in adult colouration, in cases where sexual dymorphism is very relevant app recognizes only female colouration only. 
            - Unusal poses and colouration change freshly post molt or colouration vanishing long time after molt may impact prediction accuracy.
            - Prediction accuracy is approximately 98%. If you notice above 2/100 missed prediction which is avarage fot this model, let me know.
            - For any feedback or questions contact me on Facebook (Paweł Grygielski)
            """)
        else:
            st.markdown("""
            - Prześlij zdjęcie całego pająka, zrobione z góry.
            - Aplikacja rozpoznaje tylko gatunki wypisane na liście gatunków.
            - W przypadku gatunków nieobecnych na liście, ale obecnych ich rodzajów, aplikacja najprawdopodobniej prawidłowo rozpozna rodzaj, ale przypisze gatunek do jednego z dostępnych na liście.
            - Aplikacja służy do rozpoznawaniu ptaszników o ubarwieniu osobnika dorosłego, w przypadkach gdy dymorfizm płciowy jest znaczący aplikacja rozpoznaje tylko samice.
            - Niestandardowe ustawienie pająka, zmiany kolorów ze względu na śiweżo przebytą wylinkę lub zanik kolorów ze względu na długi okres bez procesu przechodzenia wylinki mogą wpłynąć na jakość predykcji
            - Kalkulowana jakość predykcji to około 98%. Jeśli zauważysz, że aplikacja myli się znacząco częściej niż w 2/100 przypadkach, możesz to zgłosić.
            - Wszelkie zgłoszenia i pytania można wysyłać w wiadomości prywatnej na moim prywatnym Facebooku (Paweł Grygielski)
            """)

    elif page == ("Credits" if lang == "English" else "Podziękowania"):
        st.title("Credits" if lang == "English" else "Podziękowania")
        if lang == "English":
            st.markdown("""
            Thanks to the breeders for sharing their photos used for training the model:
            """)
        else:
            st.markdown("""
            Podziękowania dla hodowców za udostępnienie zdjęć wykorzystanych do treningu modelu:
            """)

        breeders = {
            "Fornal": "https://www.facebook.com/fornalpets?locale=pl_PL",
            "Spider Shop": "https://www.facebook.com/spidershoppl?locale=pl_PL",
            "SpidersOnline": "https://www.facebook.com/spidershoppl?locale=pl_PL",
            "Arent": "https://www.facebook.com/arent.spiders?locale=pl_PL",
            "Arachnohobbia": "https://www.facebook.com/profile.php?id=100064755876648&locale=pl_PL"
          }

        for name, fb_link in breeders.items():
            st.markdown(f"- [{name}]({fb_link})")

if __name__ == "__main__":
    main()
