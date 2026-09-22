import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from plant_info import get_plant_info


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LeafLens",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_PATH = os.path.join(
    "data",
    "processed",
    "cnn",
    "mobilenetv3small_finetuned_best.keras",
)

CLASS_NAMES = [
    "Aloevera",
    "Amla",
    "Amruta_Balli",
    "Arali",
    "Ashoka",
    "Ashwagandha",
    "Avacado",
    "Bamboo",
    "Basale",
    "Betel",
    "Betel_Nut",
    "Brahmi",
    "Castor",
    "Curry_Leaf",
    "Doddapatre",
    "Ekka",
    "Ganike",
    "Gauva",
    "Geranium",
    "Henna",
    "Hibiscus",
    "Honge",
    "Insulin",
    "Jasmine",
    "Lemon",
    "Lemon_grass",
    "Mango",
    "Mint",
    "Nagadali",
    "Neem",
    "Nithyapushpa",
    "Nooni",
    "Pappaya",
    "Pepper",
    "Pomegranate",
    "Raktachandini",
    "Rose",
    "Sapota",
    "Tulasi",
    "Wood_sorel",
]


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "image_source" not in st.session_state:
    st.session_state.image_source = None

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "predicted_plant" not in st.session_state:
    st.session_state.predicted_plant = None

if "prediction_unknown" not in st.session_state:
    st.session_state.prediction_unknown = False


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ============================================================
# MODEL PREDICTION
# ============================================================

def _prepare_image(image):
    """Apply the exact preprocessing used during CNN training."""
    image = image.convert("RGB")
    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )

    image_array = np.array(
        image,
        dtype=np.float32,
    )

    image_array = (image_array / 127.5) - 1.0
    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_array


def predict_plant(image):
    """Predict the plant using the trained CNN."""
    image_array = _prepare_image(image)

    predictions = model.predict(
        image_array,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_plant = CLASS_NAMES[predicted_index]

    return predicted_plant


def classify_plant(image):
    """Return the top CNN prediction without an additional safeguard."""
    return predict_plant(image), False


# ============================================================
# NAVIGATION
# ============================================================

def go_to(page):

    st.session_state.page = page

    if page != "image_search":
        st.session_state.image_source = None

    st.rerun()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');


/* ==========================================================
   GLOBAL
========================================================== */

.stApp {
    background: #f5f2e9;
    color: #193b27;
}

.main .block-container {
    max-width: 1180px;
    padding-top: 0.4rem;
    padding-bottom: 4rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ==========================================================
   NAVBAR
========================================================== */

.leaflens-logo {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.055em;
    color: #173d27;
}

.leaflens-logo span {
    color: #759565;
}

.nav-caption {
    text-align: right;
    padding-top: 0.7rem;
    color: #657269;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
}


/* ==========================================================
   HERO
========================================================== */

.hero {
    position: relative;
    min-height: 640px;
    width: 100%;

    border-radius: 32px;
    overflow: hidden;

    background-image:
        linear-gradient(
            90deg,
            rgba(5, 34, 18, 0.96) 0%,
            rgba(5, 34, 18, 0.88) 38%,
            rgba(5, 34, 18, 0.48) 68%,
            rgba(5, 34, 18, 0.16) 100%
        ),
        url("https://images.unsplash.com/photo-1497250681960-ef046c08a56e?auto=format&fit=crop&w=1800&q=90");

    background-size: cover;
    background-position: center;

    display: flex;
    align-items: center;

    box-shadow:
        0 25px 70px rgba(20, 55, 31, 0.20);
}

.hero-content {
    max-width: 650px;
    padding: 70px;
}

.hero-pill {
    display: inline-block;

    padding: 9px 17px;
    margin-bottom: 22px;

    border-radius: 999px;

    background: rgba(221, 238, 213, 0.12);
    border: 1px solid rgba(221, 238, 213, 0.28);

    color: #e1efd9;

    font-family: 'DM Sans', sans-serif;
    font-size: 0.74rem;
    font-weight: 700;

    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.hero-title {
    margin: 0 0 25px 0;

    color: #ffffff;

    font-family: 'Playfair Display', serif;
    font-size: clamp(4.2rem, 7vw, 7rem);
    font-weight: 600;

    line-height: 0.88;
    letter-spacing: -0.06em;
}

.hero-description {
    max-width: 570px;

    color: #edf5e9;

    font-family: 'DM Sans', sans-serif;
    font-size: 1.13rem;
    line-height: 1.75;
}

.hero-note {
    margin-top: 24px;

    color: #c7dcc0;

    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
}


/* ==========================================================
   PAGE HEADINGS
========================================================== */

.page-title {
    margin-top: 2.7rem;
    margin-bottom: 0.7rem;

    color: #193d28;

    font-family: 'Playfair Display', serif;
    font-size: clamp(3rem, 6vw, 5rem);
    font-weight: 600;

    line-height: 0.95;
    letter-spacing: -0.055em;
}

.page-subtitle {
    margin-bottom: 2.5rem;

    color: #68756c;

    font-family: 'DM Sans', sans-serif;
    font-size: 1.08rem;
    line-height: 1.6;
}


/* ==========================================================
   CARDS
========================================================== */

.search-card-content {
    min-height: 300px;

    padding: 2.2rem;

    background: #fffdf8;

    border: 1px solid #e1e4d9;
    border-radius: 28px;

    box-shadow:
        0 16px 45px rgba(35, 65, 43, 0.07);
}

.card-icon {
    font-size: 3rem;
    line-height: 1;
    margin-bottom: 1.4rem;
}

.card-title {
    color: #1b402a;

    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 600;

    line-height: 1.1;
    letter-spacing: -0.03em;

    margin-bottom: 0.9rem;
}

.card-description {
    color: #68746c;

    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;

    line-height: 1.7;
}


/* ==========================================================
   UNKNOWN / UNSUPPORTED RESULT
========================================================== */

.unknown-card {
    background: #fffdf8;

    border: 1px solid #dfe5da;

    border-radius: 28px;

    padding: 3rem 2.5rem;

    margin-top: 2rem;

    text-align: center;

    box-shadow:
        0 18px 50px rgba(35, 65, 43, 0.08);
}

.unknown-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
}

.unknown-title {
    color: #193d28;

    font-family: 'Playfair Display', serif;

    font-size: 2.7rem;
    font-weight: 600;

    line-height: 1.05;

    margin-bottom: 1rem;
}

.unknown-description {
    max-width: 650px;
    margin: 0 auto;

    color: #68746c;

    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    line-height: 1.7;
}

.unknown-note {
    max-width: 620px;
    margin: 1.5rem auto 0;

    padding: 1rem 1.2rem;

    background: #e2ecde;
    border: 1px solid #d2dfcb;
    border-radius: 18px;

    color: #34583e;

    font-family: 'DM Sans', sans-serif;
    line-height: 1.6;
}


/* ==========================================================
   RESULT
========================================================== */

.result-card {
    background: #fffdf8;

    border: 1px solid #dfe5da;

    border-radius: 28px;

    padding: 2.5rem;

    margin-top: 2rem;

    box-shadow:
        0 18px 50px rgba(35, 65, 43, 0.08);
}

.result-icon {
    font-size: 3rem;
}

.result-label {
    color: #748078;

    font-size: 0.78rem;
    font-weight: 700;

    letter-spacing: 0.14em;
    text-transform: uppercase;

    margin-top: 1rem;
}

.result-name {
    color: #183d27;

    font-family: 'Playfair Display', serif;

    font-size: 3.4rem;
    font-weight: 600;

    line-height: 1;

    margin-top: 0.5rem;
}


/* ==========================================================
   INFO BOX
========================================================== */

.info-box {
    margin-top: 1.8rem;

    padding: 1.25rem 1.5rem;

    background: #e2ecde;

    border: 1px solid #d2dfcb;

    border-radius: 20px;

    color: #34583e;

    font-family: 'DM Sans', sans-serif;

    line-height: 1.65;
}


/* ==========================================================
   BUTTONS
========================================================== */

.stButton > button {
    min-height: 3.25rem !important;

    border-radius: 999px !important;

    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;

    background: #fffdf8 !important;

    color: #2e5a3b !important;

    border: 1px solid #3b6748 !important;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        background 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;

    background: #f0f5ed !important;

    box-shadow:
        0 8px 22px rgba(35, 72, 45, 0.12) !important;
}

.stButton > button[kind="primary"] {
    background: #2f6140 !important;
    color: white !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover {
    background: #244e32 !important;
}


/* ==========================================================
   INPUTS
========================================================== */

.stTextInput > div > div > input {
    min-height: 3.2rem !important;

    background: #fffdf8 !important;

    color: #193b27 !important;

    border: 1px solid #d7ddd2 !important;

    border-radius: 14px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #5d8063 !important;

    box-shadow:
        0 0 0 2px rgba(93, 128, 99, 0.12) !important;
}

[data-testid="stFileUploader"] {
    background: #fffdf8;
    border: 1px solid #e0e4da;
    border-radius: 18px;
}


/* ==========================================================
   FOOTER
========================================================== */

.footer-text {
    margin-top: 5rem;
    padding-top: 2rem;

    text-align: center;

    border-top: 1px solid #d9ded4;

    color: #7c867e;

    font-family: 'DM Sans', sans-serif;

    font-size: 0.84rem;
}


/* ==========================================================
   MOBILE
========================================================== */

@media (max-width: 768px) {

    .leaflens-logo {
        font-size: 2.15rem;
    }

    .hero {
        min-height: 590px;
        border-radius: 26px;
        background-position: 62% center;
    }

    .hero-content {
        padding: 40px 30px;
    }

    .hero-title {
        font-size: 4.2rem;
    }

    .hero-description {
        font-size: 1rem;
    }

    .page-title {
        font-size: 3rem;
    }

    .search-card-content {
        min-height: auto;
        padding: 1.8rem;
    }

    .result-name {
        font-size: 2.8rem;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# NAVBAR
# ============================================================

col_logo, col_nav = st.columns([2, 1])

with col_logo:

    st.markdown(
        '<div class="leaflens-logo">Leaf<span>Lens</span></div>',
        unsafe_allow_html=True,
    )

with col_nav:

    st.markdown(
        '<div class="nav-caption">'
        'Explore&nbsp; • &nbsp;Identify&nbsp; • &nbsp;Learn'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 1 — WELCOME
# ============================================================

if st.session_state.page == "welcome":

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="hero">
<div class="hero-content">

<div class="hero-pill">
AI-POWERED PLANT DISCOVERY
</div>

<div class="hero-title">
Welcome to<br>LeafLens.
</div>

<div class="hero-description">
Discover plants, identify leaves, and explore useful information
about the plants around you.
</div>

<div class="hero-note">
Identify &nbsp;•&nbsp; Explore &nbsp;•&nbsp; Learn
</div>

</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.25, 1, 1.25])

    with col2:

        if st.button(
            "🌿  Get Started",
            use_container_width=True,
            type="primary",
        ):
            go_to("search")

    st.markdown(
        """
<div class="info-box">
<strong>A simpler way to explore plants.</strong><br>
Use a photograph to identify a plant or search by name
to explore plant information.
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2 — SEARCH METHOD
# ============================================================

elif st.session_state.page == "search":

    st.markdown(
        '<div class="page-title">'
        'How would you like to search?'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Choose the way you want to discover your plant.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    # IMAGE SEARCH

    with col1:

        st.markdown(
            """
<div class="search-card-content">

<div class="card-icon">
📷
</div>

<div class="card-title">
Image Based Search
</div>

<div class="card-description">
Have a leaf in front of you? Upload a photograph
and let LeafLens analyze it using computer vision.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Choose Image Search →",
            key="image_search_button",
            use_container_width=True,
            type="primary",
        ):
            go_to("image_search")

    # NAME SEARCH

    with col2:

        st.markdown(
            """
<div class="search-card-content">

<div class="card-icon">
🔎
</div>

<div class="card-title">
Name Based Search
</div>

<div class="card-description">
Already know the plant name? Search for it
and explore its information.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Choose Name Search →",
            key="name_search_button",
            use_container_width=True,
            type="primary",
        ):
            go_to("name_search")

    st.markdown(
        """
<div class="info-box">
<strong>Not sure what the plant is?</strong>
Choose Image Based Search. Already know the name?
Use Name Based Search.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "← Back to Welcome",
        key="back_to_welcome",
    ):
        go_to("welcome")


# ============================================================
# PAGE 3 — IMAGE SEARCH
# ============================================================

elif st.session_state.page == "image_search":

    st.markdown(
        '<div class="page-title">'
        'Identify a plant'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'How would you like to add your plant photo?'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SOURCE SELECTION
    # --------------------------------------------------------

    if st.session_state.image_source is None:

        col1, col2 = st.columns(2, gap="large")

        with col1:

            st.markdown(
                """
<div class="search-card-content">

<div class="card-icon">
📷
</div>

<div class="card-title">
Click a Photo
</div>

<div class="card-description">
Use your device camera to capture a clear
photograph of the leaf.
</div>

</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(
                "📷 Open Camera",
                key="open_camera",
                use_container_width=True,
                type="primary",
            ):

                st.session_state.image_source = "camera"
                st.rerun()

        with col2:

            st.markdown(
                """
<div class="search-card-content">

<div class="card-icon">
🖼️
</div>

<div class="card-title">
Upload from Gallery
</div>

<div class="card-description">
Choose an existing plant photograph
from your device.
</div>

</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(
                "🖼️ Choose from Gallery",
                key="open_gallery",
                use_container_width=True,
                type="primary",
            ):

                st.session_state.image_source = "gallery"
                st.rerun()

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    elif st.session_state.image_source == "camera":

        st.markdown(
            '<div class="card-title">📷 Take a Photo</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card-description">'
            'Position the leaf clearly inside the camera view.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        camera_image = st.camera_input(
            "Take a photo",
            key="camera_input",
        )

        if camera_image is not None:

            st.image(
                camera_image,
                caption="Selected image",
                use_container_width=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(
                "🔍 Identify This Plant",
                key="identify_camera",
                use_container_width=True,
                type="primary",
            ):

                if model is None:

                    st.error(
                        "LeafLens model could not be loaded."
                    )

                else:

                    with st.spinner(
                        "Analyzing the leaf..."
                    ):

                        image = Image.open(
                            camera_image
                        )

                        prediction, is_unknown = classify_plant(
                            image
                        )

                    st.session_state.predicted_plant = prediction
                    st.session_state.prediction_unknown = is_unknown

                    go_to("result")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "← Choose Another Option",
            key="back_camera",
        ):

            st.session_state.image_source = None
            st.rerun()

    # --------------------------------------------------------
    # GALLERY
    # --------------------------------------------------------

    elif st.session_state.image_source == "gallery":

        st.markdown(
            '<div class="card-title">'
            '🖼️ Upload from Gallery'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card-description">'
            'Choose a clear image of the plant leaf.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            key="gallery_input",
        )

        if uploaded_image is not None:

            st.image(
                uploaded_image,
                caption="Selected image",
                use_container_width=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(
                "🔍 Identify This Plant",
                key="identify_gallery",
                use_container_width=True,
                type="primary",
            ):

                if model is None:

                    st.error(
                        "LeafLens model could not be loaded."
                    )

                else:

                    with st.spinner(
                        "Analyzing the leaf..."
                    ):

                        image = Image.open(
                            uploaded_image
                        )

                        prediction, is_unknown = classify_plant(
                            image
                        )

                    st.session_state.predicted_plant = prediction
                    st.session_state.prediction_unknown = is_unknown

                    go_to("result")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "← Choose Another Option",
            key="back_gallery",
        ):

            st.session_state.image_source = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "← Back to Search",
        key="back_image_search",
    ):

        go_to("search")


# ============================================================
# PAGE 4 — NAME SEARCH
# ============================================================

elif st.session_state.page == "name_search":

    st.markdown(
        '<div class="page-title">'
        'Search by plant name'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Enter the name of a plant to explore its information.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="search-card-content">

<div class="card-icon">
🌿
</div>

<div class="card-title">
Find a plant
</div>

<div class="card-description">
Enter the plant name below and explore its information.
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    plant_name = st.text_input(
        "Plant name",
        placeholder="e.g. Mint, Neem, Tulasi...",
    )

    if st.button(
        "🌿 Get Plant Information",
        use_container_width=True,
        type="primary",
    ):

        if plant_name.strip():

            entered_name = plant_name.strip().lower().replace("_", " ")
            matched_key = None

            for class_name in CLASS_NAMES:

                info = get_plant_info(class_name)

                if info:
                    database_name = info["display_name"].lower().replace(
                        "_", " ")
                    class_display = class_name.lower().replace("_", " ")

                    if entered_name == database_name or entered_name == class_display:
                        matched_key = class_name
                        break

            if matched_key:
                st.session_state.predicted_plant = matched_key
                st.session_state.prediction_unknown = False
                go_to("result")

            else:
                st.error(
                    "This plant is not currently available in the LeafLens database. "
                    "Please try one of the supported plants."
                )

        else:
            st.warning("Please enter a plant name.")

    st.markdown(
        """
<div class="info-box">
<strong>Explore plants.</strong>
Search from LeafLens' supported plant classes to view their
description, scientific name, characteristics, habitat, uses,
and an interesting fact.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "← Back to Search",
        key="back_name_search",
    ):
        go_to("search")


# ============================================================
# PAGE 5 — RESULT
# ============================================================

elif st.session_state.page == "result":

    plant_name = st.session_state.predicted_plant

    st.markdown(
        '<div class="page-title">'
        'Plant identified'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore the plant information available in LeafLens.'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.prediction_unknown:

        st.markdown(
            """
<div class="unknown-card">

<div class="unknown-icon">🌿</div>

<div class="unknown-title">
Plant not confidently recognized
</div>

<div class="unknown-description">
LeafLens could not confidently match this image to one of its
supported plant classes. Try a clear photo focused on the leaf
with good lighting and minimal background.
</div>

<div class="unknown-note">
<strong>LeafLens currently supports 40 plant classes.</strong><br>
If the plant is outside the supported classes, the app will ask
you to try another image instead of presenting an uncertain
identification as a result.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    elif plant_name:

        plant_info = get_plant_info(plant_name)

        if plant_info:

            display_name = plant_info["display_name"]
            scientific_name = plant_info["scientific_name"]
            description = plant_info["description"]
            plant_type = plant_info["plant_type"]
            leaf_features = plant_info["leaf_features"]
            flowers_or_fruit = plant_info["flowers_or_fruit"]
            habitat = plant_info["habitat"]
            common_uses = plant_info["common_uses"]
            interesting_fact = plant_info["interesting_fact"]

            # ----------------------------------------------------
            # IDENTIFICATION CARD
            # ----------------------------------------------------

            st.markdown(
                f"""
<div class="result-card">

<div class="result-icon">
🌿
</div>

<div class="result-label">
LeafLens identification
</div>

<div class="result-name">
{display_name}
</div>

<div style="
    color:#6f7d73;
    font-family:'DM Sans', sans-serif;
    font-size:1.05rem;
    margin-top:1rem;
    font-style:italic;
">
{scientific_name}
</div>

</div>
""",
                unsafe_allow_html=True,
            )

            # ----------------------------------------------------
            # ABOUT
            # ----------------------------------------------------

            st.markdown(
                f"""
<div class="search-card-content" style="margin-top:2rem;">

<div class="card-title">
About {display_name}
</div>

<div class="card-description">
{description}
</div>

<br>

<div style="
    color:#748078;
    font-size:0.78rem;
    font-weight:700;
    letter-spacing:0.14em;
    text-transform:uppercase;
">
Scientific name
</div>

<div style="
    color:#1b402a;
    font-family:'Playfair Display', serif;
    font-size:1.55rem;
    margin-top:0.4rem;
">
<i>{scientific_name}</i>
</div>

</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ----------------------------------------------------
            # PLANT CHARACTERISTICS
            # ----------------------------------------------------

            st.markdown(
                '<div class="card-title" style="margin-top:1rem;">'
                'Plant characteristics'
                '</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown(
                    f"""
<div class="search-card-content" style="min-height:220px;">

<div style="font-size:2rem; margin-bottom:0.8rem;">🌱</div>

<div class="card-title" style="font-size:1.45rem;">
Plant type
</div>

<div class="card-description">
{plant_type}
</div>

<br>

<div style="font-size:2rem; margin-bottom:0.8rem;">🍃</div>

<div class="card-title" style="font-size:1.45rem;">
Leaf features
</div>

<div class="card-description">
{leaf_features}
</div>

</div>
""",
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
<div class="search-card-content" style="min-height:220px;">

<div style="font-size:2rem; margin-bottom:0.8rem;">🌸</div>

<div class="card-title" style="font-size:1.45rem;">
Flowers & fruit
</div>

<div class="card-description">
{flowers_or_fruit}
</div>

<br>

<div style="font-size:2rem; margin-bottom:0.8rem;">🌍</div>

<div class="card-title" style="font-size:1.45rem;">
Habitat & growth
</div>

<div class="card-description">
{habitat}
</div>

</div>
""",
                    unsafe_allow_html=True,
                )

            # ----------------------------------------------------
            # COMMON USES
            # ----------------------------------------------------

            st.markdown(
                f"""
<div class="info-box">

<strong>Common uses</strong>

{common_uses}

</div>
""",
                unsafe_allow_html=True,
            )

            # ----------------------------------------------------
            # INTERESTING FACT
            # ----------------------------------------------------

            st.markdown(
                f"""
<div class="info-box" style="
    margin-top:1rem;
    background:#f1eee3;
    border-color:#ddd7c5;
">

<strong>🌿 Did you know?</strong>

{interesting_fact}

</div>
""",
                unsafe_allow_html=True,
            )

        else:

            st.warning(
                "Information for this plant is not currently available "
                "in the LeafLens database."
            )

    else:

        st.warning("No plant has been selected yet.")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🔍 Identify Another Plant",
        use_container_width=True,
        type="primary",
    ):
        go_to("image_search")

    if st.button(
        "← Back to Search",
        key="result_back_search",
    ):
        go_to("search")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer-text">
🌿 LeafLens • Plant Identification & Information
</div>
""",
    unsafe_allow_html=True,
)
