import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import streamlit.components.v1 as components

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

def tarantupedia_link(name):
    """Generuje link do tarantupedia.com"""
    parts = name.lower().split()
    if len(parts) == 1:
        return f"https://www.tarantupedia.com/theraphosinae/{parts[0]}"
    else:
        genus = parts[0]
        species = '-'.join(parts)
        return f"https://www.tarantupedia.com/theraphosinae/{genus}/{species}"

# Load model at startup
model = load_trained_model()

# Set session state for navigation and language if not already set
if 'page' not in st.session_state:
    st.session_state['page'] = 'prediction'
    
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'English'

# Function to change page
def change_page(page_name):
    st.session_state['page'] = page_name
    
# Function to change language
def change_language(language):
    st.session_state['lang'] = language

# Custom CSS and JS for the app
def inject_custom_css_and_js():
    # Custom CSS for the app
    st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #1c1c1c 0%, #2d2d2d 100%);
        color: #f0f0f0;
    }
    
    /* Custom container for the app content */
    .main-container {
        background-color: rgba(15, 15, 15, 0.7);
        border-radius: 15px;
        padding: 20px;
        margin-top: 60px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling */
    .app-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 60px;
        background-color: rgba(10, 10, 10, 0.9);
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 20px;
        z-index: 1000;
        backdrop-filter: blur(5px);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    /* Hamburger menu styling */
    .hamburger-menu {
        width: 30px;
        height: 30px;
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        cursor: pointer;
    }
    
    .hamburger-line {
        width: 100%;
        height: 3px;
        background-color: #f0f0f0;
        border-radius: 3px;
        transition: all 0.3s ease;
    }
    
    /* App title styling */
    .app-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #f0f0f0;
        display: flex;
        align-items: center;
    }
    
    .app-title span {
        margin-left: 10px;
    }
    
    /* Language selector styling */
    .language-selector {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .language-btn {
        padding: 5px 10px;
        background: none;
        border: 1px solid #f0f0f0;
        border-radius: 5px;
        color: #f0f0f0;
        cursor: pointer;
        font-size: 0.8rem;
        transition: all 0.3s ease;
    }
    
    .language-btn.active {
        background-color: #f0f0f0;
        color: #1c1c1c;
    }
    
    /* Navigation menu styling */
    .nav-menu {
        position: fixed;
        top: 60px;
        left: -250px;
        width: 250px;
        height: calc(100% - 60px);
        background-color: rgba(20, 20, 20, 0.95);
        transition: left 0.3s ease;
        z-index: 999;
        backdrop-filter: blur(5px);
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.2);
        overflow-y: auto;
    }
    
    .nav-menu.show {
        left: 0;
    }
    
    .nav-menu-item {
        padding: 15px 20px;
        border-bottom: 1px solid #333;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
    }
    
    .nav-menu-item:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    .nav-menu-item.active {
        background-color: rgba(255, 255, 255, 0.2);
        border-left: 4px solid #f0f0f0;
    }
    
    .nav-menu-item i {
        margin-right: 10px;
    }
    
    /* Overlay when menu is open */
    .overlay {
        position: fixed;
        top: 60px;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        display: none;
        z-index: 998;
    }
    
    .overlay.show {
        display: block;
    }
    
    /* Species card styling */
    .species-card {
        background-color: rgba(30, 30, 30, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .species-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #8c1aff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        border-radius: 30px;
    }
    
    .stButton>button:hover {
        background-color: #7209db;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #8c1aff;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        background-color: rgba(140, 26, 255, 0.1);
    }
    
    /* Prediction result styling */
    .prediction-result {
        background-color: rgba(30, 30, 30, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Tarantula animation */
    .tarantula-logo {
        font-size: 24px;
        animation: crawl 5s infinite alternate;
    }
    
    @keyframes crawl {
        0% { transform: translateX(0); }
        100% { transform: translateX(20px); }
    }
    
    /* Background spider web effect */
    .web-background {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("https://i.imgur.com/DBGp1Yv.png");
        background-size: cover;
        opacity: 0.1;
        pointer-events: none;
        z-index: -1;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1c1c1c;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #8c1aff;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #7209db;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Make the prediction image look better */
    .uploaded-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        max-height: 300px;
        object-fit: contain;
    }
    
    /* Species list grid */
    .species-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 10px;
        margin-top: 20px;
    }
    
    /* Usage instructions styling */
    .usage-card {
        background-color: rgba(30, 30, 30, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #8c1aff;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #8c1aff !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .app-title {
            font-size: 1rem;
        }
        
        .species-grid {
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Custom JavaScript for the app
    st.markdown("""
    <script>
    // Function to toggle the navigation menu
    function toggleMenu() {
        const navMenu = document.querySelector('.nav-menu');
        const overlay = document.querySelector('.overlay');
        
        navMenu.classList.toggle('show');
        overlay.classList.toggle('show');
    }
    
    // Function to close the menu when clicking outside
    function closeMenuOnClickOutside() {
        const overlay = document.querySelector('.overlay');
        if (overlay) {
            overlay.addEventListener('click', function() {
                const navMenu = document.querySelector('.nav-menu');
                navMenu.classList.remove('show');
                overlay.classList.remove('show');
            });
        }
    }
    
    // Execute after the DOM is fully loaded
    document.addEventListener('DOMContentLoaded', function() {
        // Add click event to hamburger menu
        const hamburger = document.querySelector('.hamburger-menu');
        if (hamburger) {
            hamburger.addEventListener('click', toggleMenu);
        }
        
        closeMenuOnClickOutside();
    });
    </script>
    """, unsafe_allow_html=True)

def render_app_header():
    # App header with hamburger menu and language selector
    header_html = f"""
    <div class="app-header">
        <div class="hamburger-menu" onclick="toggleMenu()">
            <div class="hamburger-line"></div>
            <div class="hamburger-line"></div>
            <div class="hamburger-line"></div>
        </div>
        
        <div class="app-title">
            <div class="tarantula-logo">🕷️</div>
            <span>{'Theraphosidae Species Classifier' if st.session_state['lang'] == 'English' else 'Klasyfikator gatunków Theraphosidae'}</span>
        </div>
        
        <div class="language-selector">
            <button class="language-btn {'active' if st.session_state['lang'] == 'English' else ''}" 
                    onclick="window.parent.postMessage({{action: 'changeLanguage', language: 'English'}}, '*')">EN</button>
            <button class="language-btn {'active' if st.session_state['lang'] == 'Polski' else ''}" 
                    onclick="window.parent.postMessage({{action: 'changeLanguage', language: 'Polski'}}, '*')">PL</button>
        </div>
    </div>
    
    <div class="nav-menu">
        <div class="nav-menu-item {'active' if st.session_state['page'] == 'prediction' else ''}" 
             onclick="window.parent.postMessage({{action: 'changePage', page: 'prediction'}}, '*')">
            <i>🔎</i> {'Prediction' if st.session_state['lang'] == 'English' else 'Predykcja'}
        </div>
        <div class="nav-menu-item {'active' if st.session_state['page'] == 'species_list' else ''}" 
             onclick="window.parent.postMessage({{action: 'changePage', page: 'species_list'}}, '*')">
            <i>📋</i> {'Species List' if st.session_state['lang'] == 'English' else 'Lista gatunków'}
        </div>
        <div class="nav-menu-item {'active' if st.session_state['page'] == 'usage' else ''}" 
             onclick="window.parent.postMessage({{action: 'changePage', page: 'usage'}}, '*')">
            <i>📖</i> {'Usage' if st.session_state['lang'] == 'English' else 'Instrukcja'}
        </div>
    </div>
    
    <div class="overlay" onclick="toggleMenu()"></div>
    
    <div class="web-background"></div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)
    
    # JavaScript to handle message events for navigation and language change
    components.html("""
    <script>
    window.addEventListener('message', function(event) {
        const data = event.data;
        
        if (data.action === 'changePage') {
            // Send to Streamlit
            const streamlit = window.parent.window.streamlit;
            if (streamlit) {
                streamlit.setComponentValue({
                    type: 'page_change',
                    page: data.page
                });
            }
        }
        
        if (data.action === 'changeLanguage') {
            // Send to Streamlit
            const streamlit = window.parent.window.streamlit;
            if (streamlit) {
                streamlit.setComponentValue({
                    type: 'language_change',
                    language: data.language
                });
            }
        }
    });
    </script>
    """, height=0)


def render_prediction_page():
    st.markdown("""
    <div class="main-container">
        <h1>🔎 {'Prediction' if st.session_state['lang'] == 'English' else 'Predykcja'}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload an image (top view of full spider)" if st.session_state['lang'] == 'English' else "Prześlij zdjęcie (cały pająk, widok z góry)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner('🕷️ ' + ('Analyzing spider...' if st.session_state['lang'] == 'English' else 'Analizowanie pająka...')):
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_label = class_labels[predicted_index]
            confidence = float(predictions[0][predicted_index]) * 100

        url = tarantupedia_link(predicted_label)
        
        st.markdown("""
        <div class="prediction-result">
            <h3>✅ {'Prediction Result' if st.session_state['lang'] == 'English' else 'Wynik predykcji'}</h3>
        """, unsafe_allow_html=True)
        
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        st.markdown(f"""
            <div style="margin-top: 20px;">
                <h2 style="color: #8c1aff;">{predicted_label}</h2>
                <div style="background-color: rgba(140, 26, 255, 0.2); 
                            border-radius: 5px; 
                            padding: 5px 10px; 
                            display: inline-block;">
                    {'Confidence' if st.session_state['lang'] == 'English' else 'Pewność'}: {confidence:.2f}%
                </div>
                <p style="margin-top: 15px;">
                    <a href="{url}" target="_blank" style="color: #8c1aff; text-decoration: none;">
                        {'🔗 Learn more about this species on Tarantupedia' if st.session_state['lang'] == 'English' else '🔗 Dowiedz się więcej o tym gatunku na Tarantupedia'}
                    </a>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_species_list_page():
    st.markdown("""
    <div class="main-container">
        <h1>📋 {'Species List' if st.session_state['lang'] == 'English' else 'Lista gatunków'}</h1>
        <p>{'The model recognizes the following species and genera (mainly from the pet trade):' if st.session_state['lang'] == 'English' else 'Model rozpoznaje następujące gatunki i rodzaje (głównie popularne w handlu):'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a filter/search box
    search_term = st.text_input(
        'Search species' if st.session_state['lang'] == 'English' else 'Wyszukaj gatunek', 
        placeholder='Type to filter...' if st.session_state['lang'] == 'English' else 'Wpisz aby filtrować...'
    )
    
    # Filter the species list
    filtered_species = [species for species in class_labels if search_term.lower() in species.lower()] if search_term else class_labels
    
    # Create species grid with cards
    species_html = '<div class="species-grid">'
    for species in filtered_species:
        url = tarantupedia_link(species)
        species_html += f"""
        <a href="{url}" target="_blank" style="text-decoration: none; color: inherit;">
            <div class="species-card">
                <h4>{species}</h4>
            </div>
        </a>
        """
    species_html += '</div>'
    
    st.markdown(species_html, unsafe_allow_html=True)


def render_usage_page():
    st.markdown("""
    <div class="main-container">
        <h1>📖 {'Usage Instructions' if st.session_state['lang'] == 'English' else 'Instrukcja użycia'}</h1>

        <div class="usage-card">
            <h3>{'How to Use' if st.session_state['lang'] == 'English' else 'Jak używać'}</h3>
            <ul>
                <li>{'Upload a photo of the full spider, taken from above.' if st.session_state['lang'] == 'English' else 'Prześlij zdjęcie całego pająka, zrobione z góry.'}</li>
                <li>{'The app recognizes only the species listed in the Species List tab.' if st.session_state['lang'] == 'English' else 'Aplikacja rozpoznaje tylko gatunki wypisane na liście gatunków.'}</li>
                <li>{'If the species is not in the list but its genus is, the app will likely identify the genus correctly but assign the species to one from the list.' if st.session_state['lang'] == 'English' else 'W przypadku gatunków nieobecnych na liście, ale obecnych ich rodzajów, aplikacja najprawdopodobniej prawidłowo rozpozna rodzaj, ale przypisze gatunek do jednego z dostępnych na liście.'}</li>
            </ul>
        </div>

        <div class="usage-card">
            <h3>{'Best Practices' if st.session_state['lang'] == 'English' else 'Najlepsze praktyki'}</h3>
            <ul>
                <li>{'Use well-lit photos with a clear view of the spider.' if st.session_state['lang'] == 'English' else 'Używaj dobrze oświetlonych zdjęć z wyraźnym widokiem pająka.'}</li>
                <li>{'Avoid blurry images or photos with multiple spiders.' if st.session_state['lang'] == 'English' else 'Unikaj rozmazanych zdjęć lub zdjęć z wieloma pająkami.'}</li>
                <li>{'For best results, position the camera directly above the spider.' if st.session_state['lang'] == 'English' else 'Dla najlepszych wyników ustaw aparat bezpośrednio nad pająkiem.'}</li>
            </ul>
        </div>

        <div class="usage-card">
            <h3>{'Limitations' if st.session_state['lang'] == 'English' else 'Ograniczenia'}</h3>
            <ul>
                <li>{'The app may have difficulty identifying juvenile spiders or specimens that have recently molted.' if st.session_state['lang'] == 'English' else 'Aplikacja może mieć trudności z identyfikacją młodych pająków lub osobników, które niedawno zrzuciły wylinki.'}</li>
                <li>{'Unusual poses or partially visible spiders may reduce accuracy.' if st.session_state['lang'] == 'English' else 'Nietypowe pozy lub częściowo widoczne pająki mogą zmniejszyć dokładność.'}</li>
                <li>{'Species not in the training dataset will be matched to the most similar known species.' if st.session_state['lang'] == 'English' else 'Gatunki nieobecne w zbiorze treningowym będą dopasowane do najbardziej podobnego znanego gatunku.'}</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Set page config
    st.set_page_config(
        page_title="Tarantula Classifier",
        page_icon="🕷️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject custom CSS and JavaScript
    inject_custom_css_and_js()
    
    # Render the app header with navigation
    render_app_header()
    
    # Handle component events
    component_value = st.session_state.get('component_value', None)
    if component_value:
        if component_value.get('type') == 'page_change':
            st.session_state['page'] = component_value.get('page')
        elif component_value.get('type') == 'language_change':
            st.session_state['lang'] = component_value.get('language')
    
    # Render the appropriate page based on navigation state
    if st.session_state['page'] == 'prediction':
        render_prediction_page()
    elif st.session_state['page'] == 'species_list':
        render_species_list_page()
    else:  # usage page
        render_usage_page()
    
    # Hide Streamlit branding
    hide_st_style = """
    <style>
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
