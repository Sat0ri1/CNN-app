import streamlit as st
import streamlit.components.v1 as components
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown

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

def tarantupedia_link(name):
    parts = name.lower().split()
    if len(parts) == 1:
        return f"https://www.tarantupedia.com/theraphosinae/{parts[0]}"
    else:
        genus = parts[0]
        species = '-'.join(parts)
        return f"https://www.tarantupedia.com/theraphosinae/{genus}/{species}"

def render_top_bar(page, lang):
    html = f"""
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600&display=swap" rel="stylesheet">

    <style>
    /* Reset & fonts */
    .topbar {{
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 56px;
        background: linear-gradient(90deg, #121212cc 0%, #1a1a1add 100%);
        color: #eee;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 0 24px;
        gap: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.8);
        z-index: 9999;
        user-select: none;
    }}

    select {{
        background-color: #222;
        color: #66aaff;
        border: none;
        border-radius: 8px;
        padding: 6px 16px;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }}
    select:hover {{
        background-color: #333;
    }}
    select:focus {{
        outline: 2px solid #66aaff;
        outline-offset: 2px;
    }}

    .dropdown {{
        position: relative;
    }}

    #menu-toggle {{
        font-size: 26px;
        color: #66aaff;
        cursor: pointer;
        user-select: none;
        padding: 4px 10px;
        border-radius: 6px;
        transition: background-color 0.25s ease;
    }}
    #menu-toggle:hover {{
        background-color: #334455;
    }}

    .dropdown-content {{
        position: absolute;
        top: 56px;
        right: 0;
        background: #222;
        border: 1px solid #444;
        border-radius: 8px;
        min-width: 220px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.7);
        opacity: 0;
        visibility: hidden;
        transform: translateY(-10px);
        transition: opacity 0.25s ease, transform 0.25s ease, visibility 0.25s;
        z-index: 10000;
    }}

    .dropdown.show .dropdown-content {{
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
    }}

    .dropdown-content a {{
        display: flex;
        align-items: center;
        gap: 12px;
        color: #66aaff;
        text-decoration: none;
        padding: 14px 24px;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 6px;
        transition: background-color 0.2s ease;
    }}
    .dropdown-content a:hover {{
        background-color: #445566;
        color: #aaddff;
    }}

    .dropdown-content a i {{
        width: 20px;
        text-align: center;
    }}

    /* Spacer so main content not under fixed topbar */
    .topbar-spacer {{
        height: 56px;
        user-select: none;
    }}

    /* Scrollbar customization */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: #121212;
    }}
    ::-webkit-scrollbar-thumb {{
        background-color: #66aaff;
        border-radius: 20px;
        border: 2px solid #121212;
    }}

    /* Responsive */
    @media (max-width: 480px) {{
        .topbar {{
            justify-content: center;
            gap: 12px;
            padding: 0 12px;
        }}
        select {{
            padding: 6px 10px;
            font-size: 0.9rem;
        }}
        .dropdown-content {{
            min-width: 180px;
        }}
    }}

    </style>

    <div class="topbar" role="banner">
        <select id="lang-select" aria-label="Select Language" title="Select Language">
            <option value="en" {"selected" if lang=="en" else ""}>English</option>
            <option value="pl" {"selected" if lang=="pl" else ""}>Polski</option>
        </select>

        <div class="dropdown" id="menu-dropdown">
            <span id="menu-toggle" role="button" aria-haspopup="true" aria-expanded="false" tabindex="0" title="Toggle Menu">
                <i class="fa fa-bars" aria-hidden="true"></i>
            </span>
            <nav class="dropdown-content" role="menu" aria-label="Page Selection Menu">
                <a href="#" data-page="prediction" role="menuitem" tabindex="0" title="Prediction Page">
                    <i class="fa fa-spider"></i> {'Prediction' if lang=='en' else 'Predykcja'}
                </a>
                <a href="#" data-page="species" role="menuitem" tabindex="0" title="Species List">
                    <i class="fa fa-list"></i> {'Species List' if lang=='en' else 'Lista gatunków'}
                </a>
                <a href="#" data-page="usage" role="menuitem" tabindex="0" title="Usage Instructions">
                    <i class="fa fa-book"></i> {'Usage' if lang=='en' else 'Instrukcja'}
                </a>
            </nav>
        </div>
    </div>
    <div class="topbar-spacer"></div>

    <script>
    const dropdown = document.getElementById("menu-dropdown");
    const toggle = document.getElementById("menu-toggle");
    const langSelect = document.getElementById("lang-select");
    const links = dropdown.querySelectorAll("a");

    toggle.onclick = () => {{
        const expanded = toggle.getAttribute('aria-expanded') === 'true';
        toggle.setAttribute('aria-expanded', !expanded);
        dropdown.classList.toggle("show");
    }};
    toggle.onkeydown = (e) => {{
        if(e.key === 'Enter' || e.key === ' ') {{
            e.preventDefault();
            toggle.click();
        }}
    }};

    links.forEach(link => {{
        link.onclick = (e) => {{
            e.preventDefault();
            const page = e.target.closest('a').getAttribute("data-page");
            const lang = langSelect.value;
            const url = new URL(window.location);
            url.searchParams.set('page', page);
            url.searchParams.set('lang', lang);
            window.history.pushState(null, '', url);
            window.dispatchEvent(new Event('popstate'));
            dropdown.classList.remove("show");
            toggle.setAttribute('aria-expanded', 'false');
        }};
    }});

    langSelect.onchange = () => {{
        const page = new URL(window.location).searchParams.get("page") || "prediction";
        const lang = langSelect.value;
        const url = new URL(window.location);
        url.searchParams.set('page', page);
        url.searchParams.set('lang', lang);
        window.history.pushState(null, '', url);
        window.dispatchEvent(new Event('popstate'));
    }};

    // Click outside closes menu
    document.addEventListener('click', function(event) {{
        if (!dropdown.contains(event.target) && !toggle.contains(event.target)) {{
            dropdown.classList.remove("show");
            toggle.setAttribute('aria-expanded', 'false');
        }}
    }});
    </script>
    """
    components.html(html, height=70, scrolling=False)

def main():
    st.set_page_config(page_title="Theraphosidae Classifier", layout="wide", initial_sidebar_state="collapsed")

    # Pobierz parametry zapytania
    params = st.query_params
    lang = params.get("lang", ["en"])[0]
    page = params.get("page", ["prediction"])[0]

    # Wstrzyknięcie stylów ogólnych (tu możesz zostawić markdown bo to CSS, nie JS)
    st.markdown(
        """
        <style>
        /* Background & text */
        body, .main, .block-container {
            background: linear-gradient(135deg, #121212, #1e1e1e);
            color: #eee;
            font-family: 'Montserrat', sans-serif;
            margin: 0; padding: 0 2rem;
        }
        .stButton>button {
            background-color: #334466 !important;
            color: #eee !important;
            border-radius: 8px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            transition: background-color 0.3s ease !important;
        }
        .stButton>button:hover {
            background-color: #5577cc !important;
            cursor: pointer;
        }
        a {
            color: #66aaff;
            text-decoration: none;
            font-weight: 600;
        }
        a:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
            pred_idx = np.argmax(predictions)
            confidence = predictions[0][pred_idx]
            pred_label = class_labels[pred_idx]

            st.image(uploaded_file, caption=f"{pred_label}", use_column_width=True)
            st.write(
                f"{'Predicted species:' if lang == 'en' else 'Przewidywany gatunek:'} **{pred_label}**"
            )
            st.write(f"{'Confidence:' if lang == 'en' else 'Pewność:'} {confidence:.2%}")
            st.markdown(f"[Tarantupedia link]({tarantupedia_link(pred_label)})")

    elif page == "species":
        st.title("Species List" if lang == "en" else "Lista gatunków")
        for species in class_labels:
            st.write(f"- {species}")

    elif page == "usage":
        st.title("Usage" if lang == "en" else "Instrukcja")
        if lang == "en":
            st.markdown("""
            - Upload a clear top-down image of the spider.
            - The model will predict the species.
            - You can view the list of supported species.
            - Use the language selector to switch languages.
            """)
        else:
            st.markdown("""
            - Prześlij wyraźne zdjęcie pająka z góry.
            - Model przewidzi gatunek.
            - Możesz zobaczyć listę wspieranych gatunków.
            - Użyj selektora języka, aby zmienić język.
            """)

if __name__ == "__main__":
    model = load_trained_model()
    main()

