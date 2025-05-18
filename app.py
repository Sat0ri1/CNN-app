import streamlit as st
import streamlit.components.v1 as components
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown

# --- Ustawienia modelu ---
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
MODEL_URL = "https://drive.google.com/uc?id=1fLsy6SAk-cGi5c06XVegJPjxfb0Bclxc"

class_labels = [
    "Aphonopelma chalcodes", "Brachypelma smithi", "Grammostola rosea"
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

# --- Helpers ---
def tarantupedia_link(name):
    parts = name.lower().split()
    if len(parts) == 1:
        return f"https://www.tarantupedia.com/theraphosinae/{parts[0]}"
    genus, species = parts[0], '-'.join(parts)
    return f"https://www.tarantupedia.com/theraphosinae/{genus}/{species}"

# --- Iniekcja CSS/JS (topbar) ---
def inject_topbar_assets():
    html = """
    <!-- Font Awesome & Montserrat -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600&display=swap" rel="stylesheet">

    <style>
      /* Ukryj domyślny Streamlit header */
      header.css-2trqyj.egzxvld1 {
          display: none;
      }
      /* Centrowanie i odstęp pod nasz topbar */
      .block-container {
          max-width: 800px;
          margin: 0 auto;
          padding-top: 120px;
      }
      /* Nasz topbar */
      .custom-topbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 56px;
        background: linear-gradient(90deg, #121212cc 0%, #1a1a1add 100%);
        color: #eee;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding: 0 24px;
        gap: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.8);
        z-index: 9999;
      }
      #menu-toggle { margin-left: 12px; cursor: pointer; }
      .dropdown-content {
        display: none;
        position: fixed;
        top: 56px;
        right: 24px;
        background: #222;
        border: 1px solid #444;
        border-radius: 8px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.7);
      }
      .dropdown-content a {
        display: block;
        padding: 12px 24px;
        color: #66aaff;
        text-decoration: none;
      }
      .dropdown-content a:hover { background: #445566; }
    </style>

    <div class="custom-topbar">
      <select id="lang-select" aria-label="Select Language" title="Select Language">
        <option value="en">English</option>
        <option value="pl">Polski</option>
      </select>
      <span id="menu-toggle" role="button" aria-haspopup="true" aria-expanded="false" tabindex="0">
        <i class="fa fa-bars"></i>
      </span>
      <nav id="menu-dropdown" class="dropdown-content" role="menu" aria-label="Page Menu">
        <a href="?page=prediction"><i class="fa fa-spider"></i> Prediction</a>
        <a href="?page=species"><i class="fa fa-list"></i> Species List</a>
        <a href="?page=usage"><i class="fa fa-book"></i> Usage</a>
      </nav>
    </div>

    <script>
      const toggle = document.getElementById('menu-toggle');
      const menu = document.getElementById('menu-dropdown');
      const langSelect = document.getElementById('lang-select');
      // Przełączanie menu
      toggle.onclick = () => menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
      // Ustaw wartość select na podstawie URL
      langSelect.value = new URLSearchParams(window.location.search).get('lang') || 'en';
      // OnChange → zmiana URL
      langSelect.onchange = () => {
          const search = new URLSearchParams(window.location.search);
          search.set('lang', langSelect.value);
          window.location.search = search.toString();
      };
    </script>
    """
    components.html(html, height=0, scrolling=True)

# --- Główna aplikacja ---
def main():
    st.set_page_config(page_title="Theraphosidae Classifier", layout="wide", initial_sidebar_state="collapsed")

    # Wstrzyknięcie topbaru
    inject_topbar_assets()

    # Parametry z URL
    params = st.query_params
    lang = params.get('lang', ['en'])[0]
    page = params.get('page', ['prediction'])[0]

    # Styl globalny
    st.markdown(
        """
        <style>
        body, .main, .block-container {
            background: linear-gradient(135deg, #121212, #1e1e1e);
            color: #eee;
        }
        .stButton>button {
            background-color: #334466 !important;
            color: #eee !important;
            border-radius: 8px !important;
            padding: 8px 18px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Strony
    if page == 'prediction':
        st.title('🕷️ Klasyfikator gatunków Theraphosidae' if lang == 'pl' else '🕷️ Theraphosidae Species Classifier')
        uploaded = st.file_uploader(
            'Prześlij zdjęcie (widok z góry)' if lang == 'pl' else 'Upload an image (top view)',
            type=['jpg', 'jpeg', 'png']
        )
        if uploaded:
            img = image.load_img(uploaded, target_size=(224,224))
            arr = np.expand_dims(image.img_to_array(img),0)/255.0
            model = load_trained_model()
            preds = model.predict(arr)[0]
            idx = np.argmax(preds)
            label = class_labels[idx]
            conf = preds[idx]
            st.image(uploaded, caption=label, use_column_width=False)
            st.write(f"{'Pewność:' if lang == 'pl' else 'Confidence:'} {conf:.1%}")
            st.markdown(f"[Tarantupedia]({tarantupedia_link(label)})")
    elif page == 'species':
        st.title('Lista gatunków' if lang == 'pl' else 'Species List')
        for s in class_labels:
            st.write(f"- {s}")
    else:
        st.title('Instrukcja' if lang == 'pl' else 'Usage')
        if lang == 'pl':
            st.write('- Prześlij zdjęcie pająka z góry.')
            st.write('- Model przewidzi gatunek.')
            st.write('- Zmień język i stronę z topbaru.')
        else:
            st.write('- Upload a clear top-down spider image.')
            st.write('- Model predicts the species.')
            st.write('- Change language/page via topbar.')

if __name__ == '__main__':
    main()
