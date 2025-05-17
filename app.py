import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests

MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1fLsy6SAk-cGi5c06XVegJPjxfb0Bclxc'

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model, please wait...'):
        download_model()

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

class_labels = [
    "Acanthoscurria", "Amazonius germani", "Aphonopelma seemanni", "Augcephalus", 
    "Avicularia avicularia", "Avicularia juruensis", "Avicularia minatrix", "Avicularia purpurea", 
    "Birupes simoroxigorum", "Brachypelma albiceps", "Brachypelma auratum", "Brachypelma baumgarteni", 
    "Brachypelma boehmei", "Brachypelma emilia", "Brachypelma hamorii or smithi", "Brachypelma klaasi", 
    "Bumba horrida or tapajos", "Caribena laeta", "Caribena versicolor", "Ceratogyrus brachycephalus", 
    "Ceratogyrus darlingi", "Ceratogyrus marshalli", "Ceratogyrus meridionalis", "Ceratogyrus sanderi", 
    "Chilobrachys dyscolus", "Chilobrachys fimbriatus", "Chilobrachys huahini", "Chilobrachys natanicharum", 
    "Chromatopelma cyaneopubescens", "Cilantica devamatha", "Citharacanthus cyaneus", "Cyriocosmus aueri or bertae", 
    "Cyriocosmus bicolor", "Cyriocosmus elegans", "Cyriocosmus leetzi", "Cyriocosmus perezmilesi", "Cyriocosmus ritae", 
    "Cyriopagopus (albostriatus, longipes, minax, paganus or vonwrithi", "Cyriopagopus hainanus", "Cyriopagopus lividus", 
    "Cyriopagopus schmidti", "Davus", "Dolichothele diamantinensis", "Encyocratella olivacea", "Ephebopus cyanognathus", 
    "Ephebopus murinus", "Eucratoscelus pachypus", "Grammostola iheringi or actaeon", "Grammostola pulchra", 
    "Grammostola pulchripes", "Grammostola rosea", "Hapalopus", "Haplocosmia himalayana", "Harpactira cafreriana", 
    "Harpactira pulchripes", "Heteroscodra maculata", "Heterothele gabonensis", "Holothele longipes", "Homoeomma", 
    "Hysterocrates", "Idiothele mira", "Kochiana brunnipes", "Lampropelma nigerrimum or Phormingochilus arboricola", 
    "Lasiocyano sazimai", "Lasiodora", "Megaphobema robustum", "Monocentropus balfouri", "Neoholothele incei", 
    "Nhandu coloratovillosus", "Nhandu tripepii", "Omothymus schioedtei", "Omothymus violaceopes", 
    "Ornithoctonus aureotibialis", "Pamphobeteus antinous", "Pamphobeteus ultramarinus", "Pelinobus muticus", 
    "Phormictopus auratus", "Phormingochilus everetti", "Poecilotheria", "Poecilotheria formosa", "Poecilotheria metallica", 
    "Poecilotheria ornata", "Poecilotheria rufilata", "Poecilotheria subfusca", "Psalmopoeus cambridgei", 
    "Psalmopoeus irminia", "Psalmopoeus pulcher", "Psalmopoeus reduncus", "Psalmopoeus victori", "Pterinochilus lugardi", 
    "Pterinochilus murinus", "Selenobrachys philippinus", "Stromatopelma calceatum", "Tapinauchenius plumipes", 
    "Theraphosa", "Thrixopelma ockerti", "Tliltocatl albopilosus", "Tliltocatl vagans or kahlenbergi", "Typhochlaena seladonia", 
    "Vitalius chromatus", "Xenesthis immanis"
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
        return f"https://www.tarantupedia.com/theraphosinae/{parts[0]}"
    else:
        genus = parts[0]
        species = '-'.join(parts)
        return f"https://www.tarantupedia.com/theraphosinae/{genus}/{species}"

def main():
    set_bg_hack_url()
    
    lang = st.sidebar.selectbox("Language / Język", ["English", "Polski"])

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
