import streamlit as st
import joblib

# ==========================
# LOAD MODELS
# ==========================

rating_model = joblib.load('rating_prediction_model.pkl')
visit_model = joblib.load('visit_mode_classifier.pkl')
cosine_sim = joblib.load('cosine_similarity.pkl')
recommendation_df = joblib.load('recommendation_df.pkl')
indices = joblib.load('indices.pkl')

# ==========================
# PAGE TITLE
# ==========================

st.title("Tourism Experience Analytics")
st.write("Recommendation System")

page = st.sidebar.selectbox("Choose Module", ["Attraction Recommendation"])

st.markdown(
    """
    <style>
    /* ======= Netflix-Style Background ======= */
    .stApp {
        background: linear-gradient(
            to bottom right,
            rgba(0, 0, 0, 0.9),
            rgba(50, 20, 20, 0.95),
            rgba(20, 50, 50, 0.85)
        ),
        url("https://assets.nflxext.com/ffe/siteui/vlv3/0e8e5dc8-7a89-4cc1-9d39-3f9a7e9df39f/9b436b73-fb2f-4b89-a4c7-53a41e3b4589/IN-en-20230925-popsignuptwoweeks-perspective_alpha_website_large.jpg");

        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }

    /* ======= Title Styling ======= */
    h1, h2, h3 {
        color: #E50914 !important; /* Netflix red */
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        font-family: serif;
        font-weight: 800;
    }

    /* ======= Text Styling ======= */
    .stSelectbox label, .stButton button, .stMarkdown, .stText {
        color: white !important;
        font-weight: 500;
    }

    /* ======= Button Styling ======= */
    .stButton>button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5em 1em;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #f40612;
        transform: scale(1.05);
    }

    /* ======= Dropdown Styling ======= */
    .stSelectbox [data-baseweb="select"] {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
    }

    /* ======= Movie Title Styling ======= */
    .movie-title {
        font-size: 20px;
        color: #fff;
        font-weight: 600;
        text-shadow: 2px 2px 4px #000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# RECOMMENDATION SYSTEM
# ==================================================

if page == "Attraction Recommendation":
    st.header("Recommend Attractions")

    # Create attraction list from dataset
    attraction_list = recommendation_df['Attraction'].dropna().unique().tolist()
    attraction_list = sorted(attraction_list)

    # Dropdown menu
    attraction_name = st.selectbox("Choose Attraction", attraction_list)

    if st.button("Recommend"):
        attraction_name = attraction_name.lower().strip()
        if attraction_name in indices:
            idx = indices[attraction_name]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)[1:6]
            attraction_indices = [i[0] for i in sim_scores]
            result = recommendation_df.iloc[attraction_indices][['Attraction']]
            st.dataframe(result)
        else:
            st.error("Attraction Not Found")