import streamlit as st

def render_top_bar(page, lang):
    html = f"""
    <style>
    body {{ margin:0; padding:0; }}
    .topbar {{
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 50px;
        background: #f0f2f6;
        border-bottom: 1px solid #ccc;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 15px;
        padding: 0 20px;
        font-family: Arial, sans-serif;
        z-index: 9999;
    }}
    select {{
        font-size: 14px;
        padding: 4px;
        cursor: pointer;
    }}
    .dropdown {{
        position: relative;
        display: inline-block;
    }}
    .dropdown-content {{
        display: none;
        position: absolute;
        right: 0;
        top: 100%;
        background: white;
        border: 1px solid #ccc;
        min-width: 160px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        z-index: 10000;
    }}
    .dropdown:hover .dropdown-content {{
        display: block;
    }}
    .dropdown-content a {{
        color: #333;
        padding: 10px 15px;
        text-decoration: none;
        display: block;
    }}
    .dropdown-content a:hover {{
        background: #eee;
    }}
    </style>

    <div class="topbar">
        <form method="get" action="" style="margin:0;">
            <select name="lang" onchange="this.form.submit()">
                <option value="en" {'selected' if lang=='en' else ''}>English</option>
                <option value="pl" {'selected' if lang=='pl' else ''}>Polski</option>
            </select>
            <input type="hidden" name="page" value="{page}">
        </form>

        <div class="dropdown" style="margin-left: 10px;">
            <span style="cursor:pointer; font-weight:bold;">&#9776;</span>
            <div class="dropdown-content">
                <a href="?lang={lang}&page=prediction">{'Prediction' if lang=='en' else 'Predykcja'}</a>
                <a href="?lang={lang}&page=species">{'Species List' if lang=='en' else 'Lista gatunków'}</a>
                <a href="?lang={lang}&page=usage">{'Usage' if lang=='en' else 'Instrukcja'}</a>
            </div>
        </div>
    </div>
    <div style="height:50px;"></div>
    """
    st.html(html)

def main():
    params = st.experimental_get_query_params()
    lang = params.get("lang", ["en"])[0]
    page = params.get("page", ["prediction"])[0]

    render_top_bar(page, lang)

    st.write(f"Current page: {page}, language: {lang}")
    # Tu Twoja reszta appki (Prediction, Species List, Usage)...

if __name__ == "__main__":
    main()
