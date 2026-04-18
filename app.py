import streamlit as st
import pickle
import pandas as pd
import requests

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# -----------------------------
# Helper: Fetch movie poster
# -----------------------------
def fetch_poster(movie_id):
    """
    Returns poster URL from TMDB for a given movie_id.
    If anything fails, returns a placeholder image.
    """
    try:
        api_key = st.secrets["TMDB_API_KEY"]  # put in .streamlit/secrets.toml
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        response = requests.get(url, timeout=10)
        data = response.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster"
    except Exception:
        return "https://via.placeholder.com/500x750?text=Poster+Error"

# -----------------------------
# Load model files (cached)
# -----------------------------
@st.cache_data
def load_models():
    movies = pickle.load(open("movies_list.pkl", "rb"))
    similarity = pickle.load(open("sigmoid_kernel.pkl", "rb"))
    return movies, similarity

# -----------------------------
# Main UI
# -----------------------------
st.title("🎬 Movie Recommender System")
st.write("Select a movie and get 5 similar recommendations.")

# Try loading data with friendly error
try:
    movies, similarity = load_models()
except FileNotFoundError as e:
    st.error(
        "Required model file is missing.\n\n"
        "Make sure these files exist in the same folder as app.py:\n"
        "- movies_list.pkl\n"
        "- sigmoid_kernel.pkl"
    )
    st.stop()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Validate expected columns
required_cols = {"title", "id"}
if not required_cols.issubset(set(movies.columns)):
    st.error(f"`movies_list.pkl` must contain columns: {required_cols}")
    st.stop()

selected_movie = st.selectbox("Which movie do you like?", movies["title"].values)

def recommend(movie_title):
    idx_list = movies[movies["title"] == movie_title].index.tolist()
    if not idx_list:
        return []

    idx = idx_list[0]
    distances = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in range(1, 6):  # top 5 excluding itself
        movie_idx = distances[i][0]
        movie_id = movies.iloc[movie_idx]["id"]
        title = movies.iloc[movie_idx]["title"]
        poster = fetch_poster(movie_id)
        recommendations.append((title, poster))
    return recommendations

if st.button("Show Recommendation"):
    recs = recommend(selected_movie)

    if not recs:
        st.warning("Could not find recommendations for this movie.")
    else:
        cols = st.columns(5)
        for col, (title, poster_url) in zip(cols, recs):
            with col:
                st.text(title)
                st.image(poster_url, use_container_width=True)