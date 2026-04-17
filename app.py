import streamlit as st
import pickle
import pandas as pd
import requests

# Function to fetch poster via API
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=YOUR_API_KEY_HERE&language=en-US"
    data = requests.get(url).json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# Load the models
movies = pickle.load(open('movies_list.pkl', 'rb'))
similarity = pickle.load(open('sigmoid_kernel.pkl', 'rb'))

st.title('Movie Recommender System')

# Dropdown for movie selection
selected_movie = st.selectbox(
    'Which movie do you like?',
    movies['title'].values
)

if st.button('Show Recommendation'):
    # Logic to get recommendations
    idx = movies[movies['title'] == selected_movie].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    # Create columns for posters
    cols = st.columns(5)
    
    for i in range(1, 6):
        movie_id = movies.iloc[distances[i][0]].id
        title = movies.iloc[distances[i][0]].title
        
        with cols[i-1]:
            st.text(title)
            st.image(fetch_poster(movie_id))