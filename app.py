import ast
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .main-title { font-size: 44px; font-weight: 900; margin-bottom: 0; }
    .subtitle { font-size: 18px; color: #6B7280; margin-top: 6px; margin-bottom: 24px; }
    .section-title { font-size: 28px; font-weight: 900; margin-top: 28px; margin-bottom: 14px; }
    .metric-card {
        padding: 22px; border-radius: 18px; color: white; text-align: center;
        background: linear-gradient(135deg, #111827, #1F2937);
        border: 1px solid #374151; box-shadow: 0 8px 22px rgba(0,0,0,0.18);
    }
    .metric-value { font-size: 30px; font-weight: 900; margin-bottom: 5px; }
    .metric-label { font-size: 14px; color: #D1D5DB; }
    .project-box {
        background: #F9FAFB; border: 1px solid #E5E7EB; padding: 22px;
        border-radius: 18px; box-shadow: 0 5px 14px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    .movie-card {
        border-radius: 16px; padding: 12px; background-color: #F9FAFB;
        border: 1px solid #E5E7EB; box-shadow: 0 5px 14px rgba(0,0,0,0.08);
        margin-bottom: 18px; min-height: 100%;
    }
    .movie-title { font-weight: 800; font-size: 16px; margin-top: 8px; margin-bottom: 4px; color: #111827; }
    .movie-info { font-size: 13px; color: #374151; margin-bottom: 2px; }
    .stButton > button { width: 100%; border-radius: 12px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def file_exists(path: str) -> bool:
    return Path(path).exists()


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def parse_list_column(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def make_poster_url(row) -> str:
    poster_url = safe_text(row.get("poster_url", "")).strip()
    poster_path = safe_text(row.get("poster_path", "")).strip()
    bad_values = {"", "nan", "none", "null", "0"}

    if poster_url.lower() not in bad_values:
        if poster_url.startswith("http"):
            return poster_url
        if poster_url.startswith("/"):
            return "https://image.tmdb.org/t/p/w500" + poster_url

    if poster_path.lower() not in bad_values:
        if poster_path.startswith("http"):
            return poster_path
        if poster_path.startswith("/"):
            return "https://image.tmdb.org/t/p/w500" + poster_path
        return "https://image.tmdb.org/t/p/w500/" + poster_path

    return ""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "title" in df.columns and "original_title" not in df.columns:
        rename_map["title"] = "original_title"
    if "movie_title" in df.columns and "original_title" not in df.columns:
        rename_map["movie_title"] = "original_title"
    if "name" in df.columns and "original_title" not in df.columns:
        rename_map["name"] = "original_title"
    df = df.rename(columns=rename_map)

    defaults = {
        "original_title": "Unknown Title",
        "overview": "",
        "tags": "",
        "poster_url": "",
        "poster_path": "",
        "vote_average": 0,
        "popularity": 0,
        "release_date": "",
        "release_year": 0,
        "director": "Unknown",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df.copy())

    for col in ["original_title", "overview", "tags", "poster_url", "poster_path", "director"]:
        df[col] = df[col].fillna("").astype(str)

    df["original_title"] = df["original_title"].replace("", "Unknown Title")
    df["director"] = df["director"].replace("", "Unknown")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)

    existing_year = pd.to_numeric(df["release_year"], errors="coerce").fillna(0)
    parsed_date = pd.to_datetime(df["release_date"], errors="coerce")
    year_from_date = parsed_date.dt.year.fillna(0)
    df["release_date"] = parsed_date
    df["release_year"] = existing_year.where(existing_year > 0, year_from_date)
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

    df["poster_display_url"] = df.apply(make_poster_url, axis=1)

    for col in ["genres_list", "keywords_list", "cast_list"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_column)
        else:
            df[col] = [[] for _ in range(len(df))]

    return df


def get_all_genres(df: pd.DataFrame):
    genres = []
    if "genres_list" not in df.columns:
        return []
    for items in df["genres_list"]:
        if isinstance(items, list):
            genres.extend(items)
    return sorted(set(genres))


def display_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def no_poster_box():
    st.markdown(
        """
        <div style="width:100%; aspect-ratio:2/3; background:#E5E7EB; border:1px dashed #9CA3AF;
        border-radius:14px; display:flex; align-items:center; justify-content:center; text-align:center;
        color:#6B7280; font-weight:800; font-size:18px;">No Poster<br>Available</div>
        """,
        unsafe_allow_html=True,
    )


def display_movie_card(row, show_similarity=False):
    title = safe_text(row.get("original_title", "Unknown Title"))
    poster_url = safe_text(row.get("poster_display_url", row.get("poster_url", ""))).strip()
    rating = float(row.get("vote_average", 0) or 0)
    popularity = float(row.get("popularity", 0) or 0)
    release_year = int(row.get("release_year", 0) or 0)
    overview = safe_text(row.get("overview", ""))

    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
    if poster_url.startswith("http"):
        st.image(poster_url, use_container_width=True)
    else:
        no_poster_box()

    st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-info">⭐ Rating: {rating:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-info">🗓️ Year: {release_year if release_year > 0 else "N/A"}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-info">🔥 Popularity: {popularity:.2f}</div>', unsafe_allow_html=True)

    if show_similarity and "similarity_score" in row:
        st.markdown(f'<div class="movie-info">🎯 Similarity: {float(row["similarity_score"]):.3f}</div>', unsafe_allow_html=True)

    with st.expander("Overview"):
        st.write(overview if overview.strip() else "No overview available.")
    st.markdown("</div>", unsafe_allow_html=True)


def display_movie_grid(df: pd.DataFrame, max_columns=5):
    if df.empty:
        st.warning("No movies found.")
        return
    for start in range(0, len(df), max_columns):
        row_group = df.iloc[start:start + max_columns]
        cols = st.columns(max_columns)
        for col, (_, movie) in zip(cols, row_group.iterrows()):
            with col:
                display_movie_card(movie, show_similarity=("similarity_score" in movie.index))


def get_first_available_column(df, possible_columns):
    for col in possible_columns:
        if col in df.columns:
            return col
    return None


def explain_recommendation(selected_movie, recommended_movie, df):
    selected_rows = df[df["original_title"] == selected_movie]
    recommended_rows = df[df["original_title"] == recommended_movie]
    if selected_rows.empty or recommended_rows.empty:
        return "Explanation is not available."

    selected = selected_rows.iloc[0]
    recommended = recommended_rows.iloc[0]
    reasons = []

    common_genres = set(selected.get("genres_list", [])).intersection(set(recommended.get("genres_list", [])))
    if common_genres:
        reasons.append("shared genres like " + ", ".join(list(common_genres)[:3]))

    common_keywords = set(selected.get("keywords_list", [])).intersection(set(recommended.get("keywords_list", [])))
    if common_keywords:
        reasons.append("similar keywords like " + ", ".join(list(common_keywords)[:3]))

    common_cast = set(selected.get("cast_list", [])).intersection(set(recommended.get("cast_list", [])))
    if common_cast:
        reasons.append("shared cast members like " + ", ".join(list(common_cast)[:2]))

    selected_director = safe_text(selected.get("director", ""))
    recommended_director = safe_text(recommended.get("director", ""))
    if selected_director and selected_director != "Unknown" and selected_director == recommended_director:
        reasons.append("the same director, " + selected_director)

    if reasons:
        return "This movie is recommended because it has " + ", ".join(reasons) + "."
    return "This movie is recommended because its text features, metadata, and overall content are similar to the selected movie."


def plot_bar_chart(df, x_col, y_col, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


def filter_movies(df, search_text, genre, min_rating, year_range, sort_by):
    filtered = df.copy()
    if search_text:
        filtered = filtered[filtered["original_title"].str.contains(search_text, case=False, na=False)]
    if genre != "All":
        filtered = filtered[filtered["genres_list"].apply(lambda x: genre in x if isinstance(x, list) else False)]
    filtered = filtered[filtered["vote_average"] >= min_rating]
    if year_range is not None:
        filtered = filtered[(filtered["release_year"] >= year_range[0]) & (filtered["release_year"] <= year_range[1])]
    if sort_by in filtered.columns:
        ascending = sort_by == "original_title"
        filtered = filtered.sort_values(sort_by, ascending=ascending)
    return filtered

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_main_data():
    path = "data/content_df.csv"
    if not file_exists(path):
        return pd.DataFrame()
    return clean_data(pd.read_csv(path))


@st.cache_resource
def load_vectorizer_and_matrix(df):
    vectorizer_path = "models/count_vectorizer.pkl"
    matrix_path = "models/count_matrix.pkl"
    vectorizer = joblib.load(vectorizer_path) if file_exists(vectorizer_path) else None
    matrix = joblib.load(matrix_path) if file_exists(matrix_path) else None
    if matrix is None and vectorizer is not None and "tags" in df.columns:
        matrix = vectorizer.transform(df["tags"].fillna("").astype(str))
    return vectorizer, matrix


@st.cache_data
def load_optional_csv(path):
    if not file_exists(path):
        return pd.DataFrame()
    try:
        return clean_data(pd.read_csv(path))
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


content_df = load_main_data()

if content_df.empty:
    st.error(
        """
        data/content_df.csv was not found.

        Required structure:
        app_reccomandation system/
        ├── app.py
        ├── requirements.txt
        ├── data/content_df.csv
        └── models/count_vectorizer.pkl and models/count_matrix.pkl
        """
    )
    st.stop()

vectorizer, count_matrix = load_vectorizer_and_matrix(content_df)

try:
    cosine_sim = cosine_similarity(count_matrix, count_matrix) if count_matrix is not None else None
except Exception:
    cosine_sim = None

popular_movies_df = load_optional_csv("outputs/popular_movies_baseline.csv")
weighted_movies_df = load_optional_csv("outputs/weighted_movies_baseline.csv")
hybrid_movies_df = load_optional_csv("outputs/hybrid_ranked_movies.csv")
evaluation_df = load_optional_csv("outputs/evaluation_results.csv")
advanced_evaluation_df = load_optional_csv("outputs/advanced_evaluation_results.csv")
model_comparison_df = load_optional_csv("outputs/model_comparison_results.csv")
model_quality_summary_df = load_optional_csv("outputs/model_quality_summary.csv")

# =========================================================
# RECOMMENDER
# =========================================================
def recommend_movies(movie_title, top_n=10, min_rating=0):
    if cosine_sim is None:
        return pd.DataFrame()
    matched = content_df[content_df["original_title"] == movie_title]
    if matched.empty:
        return pd.DataFrame()
    idx = matched.index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    results = []
    for movie_idx, score in sim_scores[1:]:
        row = content_df.iloc[movie_idx].copy()
        if row["vote_average"] < min_rating:
            continue
        row["similarity_score"] = score
        results.append(row)
        if len(results) >= top_n:
            break
    return pd.DataFrame(results)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎬 Movie App")
st.sidebar.markdown("Professional Recommendation System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎯 Recommend Movies",
        "🔍 Movie Explorer",
        "🔥 Popular Movies",
        "⭐ Top Rated Movies",
        "📊 Analytics",
        "🧪 Model Evaluation",
        "ℹ️ About Project",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Info")
st.sidebar.write(f"Movies: {len(content_df):,}")
st.sidebar.write(f"Columns: {len(content_df.columns):,}")
genres = get_all_genres(content_df)
st.sidebar.write(f"Genres: {len(genres):,}")

if cosine_sim is not None:
    st.sidebar.success("Model loaded successfully.")
else:
    st.sidebar.warning("Model matrix not loaded.")

# =========================================================
# HOME PAGE
# =========================================================
if page == "🏠 Home":
    st.markdown('<div class="main-title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A professional content-based movie recommendation platform with posters, search, filters, analytics, and model evaluation.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric_card("Total Movies", f"{len(content_df):,}")
    with col2:
        display_metric_card("Average Rating", round(content_df["vote_average"].mean(), 2))
    with col3:
        display_metric_card("Total Genres", len(genres))
    with col4:
        display_metric_card("Average Popularity", round(content_df["popularity"].mean(), 2))

    st.markdown('<div class="section-title">🔥 Featured Movies</div>', unsafe_allow_html=True)
    featured = popular_movies_df.head(10) if not popular_movies_df.empty else content_df.sort_values("popularity", ascending=False).head(10)
    display_movie_grid(featured, max_columns=5)

    st.markdown('<div class="section-title">📌 About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="project-box">
        This project is a <b>Movie Recommendation System</b> built to help users discover movies similar to the ones they already like.
        It uses a <b>content-based filtering approach</b>, meaning the system recommends movies by comparing movie information such as
        genres, keywords, cast, director, and overview text.<br><br>
        The main goal of this project is to create a simple, interactive, and professional web application where users can explore
        movies, view ratings and popularity, analyze movie data, and generate personalized recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🎯 Project Objectives")
        st.markdown(
            """
            - Build a movie recommendation system using machine learning techniques.
            - Recommend similar movies based on content and metadata.
            - Create a professional Streamlit web interface.
            - Display movie posters, ratings, popularity, and release year.
            - Provide search, filtering, analytics, and evaluation pages.
            """
        )
    with col_b:
        st.markdown("### 🧠 Recommendation Technique")
        st.markdown(
            """
            - Movie features are combined into a single `tags` column.
            - Text data is converted into numerical vectors using **CountVectorizer**.
            - Similarity between movies is calculated using **Cosine Similarity**.
            - The most similar movies are recommended to the user.
            - TMDB poster links are used to display professional movie cards.
            """
        )

    st.markdown("### 🛠️ Technologies Used")
    tech1, tech2, tech3, tech4 = st.columns(4)
    tech1.success("Python")
    tech2.success("Pandas")
    tech3.success("Scikit-learn")
    tech4.success("Streamlit")
    tech5, tech6, tech7, tech8 = st.columns(4)
    tech5.info("CountVectorizer")
    tech6.info("Cosine Similarity")
    tech7.info("TMDB API")
    tech8.info("Matplotlib")

    st.markdown('<div class="section-title">🚀 App Features</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.info("🎯 Content-based movie recommendations using cosine similarity.")
    c2.info("🔍 Search and filter movies by genre, rating, year, and popularity.")
    c3.info("📊 Professional analytics and model evaluation dashboards.")

# =========================================================
# RECOMMEND PAGE
# =========================================================
elif page == "🎯 Recommend Movies":
    st.markdown('<div class="main-title">🎯 Recommend Movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Choose a movie and discover similar movies based on content features.</div>', unsafe_allow_html=True)

    if cosine_sim is None:
        st.error("Recommendation model is not loaded. Check models/count_vectorizer.pkl and models/count_matrix.pkl.")
    else:
        movie_titles = sorted(content_df["original_title"].dropna().unique())
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            selected_movie = st.selectbox("Select a movie", movie_titles)
        with col2:
            top_n = st.slider("Number of recommendations", 5, 30, 10)
        with col3:
            min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)

        selected_row = content_df[content_df["original_title"] == selected_movie].iloc[0]
        st.markdown('<div class="section-title">Selected Movie</div>', unsafe_allow_html=True)
        left, right = st.columns([1, 3])
        with left:
            poster_url = safe_text(selected_row.get("poster_display_url", "")).strip()
            if poster_url.startswith("http"):
                st.image(poster_url, use_container_width=True)
            else:
                no_poster_box()
        with right:
            st.subheader(selected_row["original_title"])
            st.write(f"⭐ Rating: {float(selected_row.get('vote_average', 0)):.2f}")
            st.write(f"🔥 Popularity: {float(selected_row.get('popularity', 0)):.2f}")
            st.write(f"🎬 Director: {selected_row.get('director', 'Unknown')}")
            if isinstance(selected_row.get("genres_list", []), list):
                st.write("🎭 Genres: " + ", ".join(selected_row.get("genres_list", [])))
            st.write(safe_text(selected_row.get("overview", "")) or "No overview available.")

        if st.button("Generate Recommendations"):
            recommendations = recommend_movies(selected_movie, top_n=top_n, min_rating=min_rating)
            st.markdown('<div class="section-title">Recommended Movies</div>', unsafe_allow_html=True)
            if recommendations.empty:
                st.warning("No recommendations found.")
            else:
                display_movie_grid(recommendations, max_columns=5)
                st.markdown('<div class="section-title">Recommendation Explanation</div>', unsafe_allow_html=True)
                movie_to_explain = st.selectbox("Choose a recommended movie to explain", recommendations["original_title"].tolist())
                st.success(explain_recommendation(selected_movie, movie_to_explain, content_df))

# =========================================================
# EXPLORER PAGE
# =========================================================
elif page == "🔍 Movie Explorer":
    st.markdown('<div class="main-title">🔍 Movie Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Search, filter, sort, and explore your full movie dataset.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search_text = st.text_input("Search movie")
    with col2:
        selected_genre = st.selectbox("Genre", ["All"] + genres)
    with col3:
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)
    with col4:
        sort_by = st.selectbox("Sort by", ["popularity", "vote_average", "release_year", "original_title"])

    valid_years = content_df[content_df["release_year"] > 0]["release_year"]
    year_range = None
    if not valid_years.empty:
        year_range = st.slider("Release year range", int(valid_years.min()), int(valid_years.max()), (int(valid_years.min()), int(valid_years.max())))

    filtered_df = filter_movies(content_df, search_text, selected_genre, min_rating, year_range, sort_by)
    st.write(f"Showing **{len(filtered_df):,}** movies")
    display_count = st.slider("Number of movies to display", 5, 100, 20)
    display_movie_grid(filtered_df.head(display_count), max_columns=5)

    with st.expander("Show table view"):
        table_columns = [col for col in ["original_title", "vote_average", "popularity", "release_year", "director"] if col in filtered_df.columns]
        st.dataframe(filtered_df[table_columns].head(200), use_container_width=True)

# =========================================================
# POPULAR PAGE
# =========================================================
elif page == "🔥 Popular Movies":
    st.markdown('<div class="main-title">🔥 Popular Movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Movies ranked by popularity.</div>', unsafe_allow_html=True)
    top_n = st.slider("Number of movies", 5, 100, 25)
    popular_display = popular_movies_df.head(top_n) if not popular_movies_df.empty else content_df.sort_values("popularity", ascending=False).head(top_n)
    display_movie_grid(popular_display, max_columns=5)
    with st.expander("Show popular movies table"):
        cols = [col for col in ["original_title", "popularity", "vote_average", "release_year"] if col in popular_display.columns]
        st.dataframe(popular_display[cols], use_container_width=True)

# =========================================================
# TOP RATED PAGE
# =========================================================
elif page == "⭐ Top Rated Movies":
    st.markdown('<div class="main-title">⭐ Top Rated Movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Movies ranked by rating, weighted score, or hybrid ranking.</div>', unsafe_allow_html=True)
    ranking_type = st.radio("Choose ranking method", ["Vote Average", "Weighted Rating", "Hybrid Ranking"], horizontal=True)
    top_n = st.slider("Number of movies", 5, 100, 25)

    if ranking_type == "Weighted Rating" and not weighted_movies_df.empty:
        display_df = weighted_movies_df.head(top_n)
    elif ranking_type == "Hybrid Ranking" and not hybrid_movies_df.empty:
        display_df = hybrid_movies_df.head(top_n)
    else:
        display_df = content_df.sort_values("vote_average", ascending=False).head(top_n)

    display_movie_grid(display_df, max_columns=5)
    with st.expander("Show ranking table"):
        cols = [col for col in ["original_title", "vote_average", "popularity", "release_year", "weighted_rating", "hybrid_score"] if col in display_df.columns]
        st.dataframe(display_df[cols], use_container_width=True)

# =========================================================
# ANALYTICS PAGE
# =========================================================
elif page == "📊 Analytics":
    st.markdown('<div class="main-title">📊 Movie Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Visual insights from the dataset.</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Genre Analysis", "Rating Analysis", "Popularity Analysis", "Year Analysis"])

    with tab1:
        st.subheader("Top Genres")
        genre_counts = {}
        for genre_list in content_df["genres_list"]:
            if isinstance(genre_list, list):
                for genre in genre_list:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        genre_df = pd.DataFrame(list(genre_counts.items()), columns=["Genre", "Count"]).sort_values("Count", ascending=False)
        if genre_df.empty:
            st.warning("No genre data available.")
        else:
            plot_bar_chart(genre_df.head(15), "Genre", "Count", "Top 15 Movie Genres", "Genre", "Number of Movies")
            st.dataframe(genre_df, use_container_width=True)

    with tab2:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(content_df["vote_average"].dropna(), bins=20)
        ax.set_title("Distribution of Movie Ratings")
        ax.set_xlabel("Vote Average")
        ax.set_ylabel("Number of Movies")
        st.pyplot(fig)
        st.subheader("Highest Rated Movies")
        st.dataframe(content_df[["original_title", "vote_average", "popularity", "release_year"]].sort_values("vote_average", ascending=False).head(30), use_container_width=True)

    with tab3:
        st.subheader("Popularity Distribution")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(content_df["popularity"].dropna(), bins=30)
        ax.set_title("Distribution of Movie Popularity")
        ax.set_xlabel("Popularity")
        ax.set_ylabel("Number of Movies")
        st.pyplot(fig)
        st.subheader("Most Popular Movies")
        st.dataframe(content_df[["original_title", "popularity", "vote_average", "release_year"]].sort_values("popularity", ascending=False).head(30), use_container_width=True)

    with tab4:
        st.subheader("Movies by Release Year")
        year_df = content_df[content_df["release_year"] > 0].groupby("release_year").size().reset_index(name="Count").sort_values("release_year")
        if year_df.empty:
            st.warning("No year data available.")
        else:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(year_df["release_year"], year_df["Count"])
            ax.set_title("Movies Released by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of Movies")
            st.pyplot(fig)
            st.dataframe(year_df, use_container_width=True)

# =========================================================
# EVALUATION PAGE
# =========================================================
elif page == "🧪 Model Evaluation":
    st.markdown('<div class="main-title">🧪 Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evaluation results and quality reports generated from your notebook.</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Evaluation Results", "Advanced Evaluation", "Model Comparison", "Quality Summary"])

    with tab1:
        st.subheader("Evaluation Results")
        st.warning("outputs/evaluation_results.csv not found.") if evaluation_df.empty else st.dataframe(evaluation_df, use_container_width=True)

    with tab2:
        st.subheader("Advanced Evaluation")
        if advanced_evaluation_df.empty:
            st.warning("outputs/advanced_evaluation_results.csv not found.")
        else:
            st.dataframe(advanced_evaluation_df, use_container_width=True)
            numeric_cols = advanced_evaluation_df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                selected_metric = st.selectbox("Select numeric metric", numeric_cols)
                label_col = get_first_available_column(advanced_evaluation_df, ["Movie", "movie", "original_title", "title", "Model", "model"])
                if label_col:
                    chart_data = advanced_evaluation_df[[label_col, selected_metric]].dropna().head(20)
                    plot_bar_chart(chart_data, label_col, selected_metric, f"{selected_metric} Comparison", label_col, selected_metric)

    with tab3:
        st.subheader("Model Comparison")
        if model_comparison_df.empty:
            st.warning("outputs/model_comparison_results.csv not found.")
        else:
            st.dataframe(model_comparison_df, use_container_width=True)
            numeric_cols = model_comparison_df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                selected_metric = st.selectbox("Select metric", numeric_cols, key="model_metric")
                label_col = get_first_available_column(model_comparison_df, ["Model", "model", "Method", "method", "Algorithm", "algorithm"])
                if label_col:
                    chart_data = model_comparison_df[[label_col, selected_metric]].dropna()
                    plot_bar_chart(chart_data, label_col, selected_metric, f"Model Comparison by {selected_metric}", "Model", selected_metric)

    with tab4:
        st.subheader("Model Quality Summary")
        st.warning("outputs/model_quality_summary.csv not found.") if model_quality_summary_df.empty else st.dataframe(model_quality_summary_df, use_container_width=True)

# =========================================================
# ABOUT PAGE
# =========================================================
elif page == "ℹ️ About Project":
    st.markdown('<div class="main-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This is a professional movie recommendation system built using Python and Streamlit.

        ### Recommendation Method
        The app uses a content-based recommendation approach. Movies are compared using their metadata and text features:
        movie overview, genres, keywords, cast, director, and combined tags.

        The text features are converted into vectors using `CountVectorizer`. Cosine similarity is then used to find movies that are most similar to the selected movie.

        ### Main Features
        - Movie recommendation engine
        - Poster-based movie cards
        - Search and filter system
        - Genre analytics
        - Rating analytics
        - Popularity analytics
        - Model evaluation dashboard
        - Professional multi-page Streamlit interface
        """
    )

    st.markdown("### Dataset Columns")
    column_summary = pd.DataFrame({"Column": content_df.columns, "Data Type": content_df.dtypes.astype(str).values, "Missing Values": content_df.isna().sum().values})
    st.dataframe(column_summary, use_container_width=True)

    st.markdown("### Required Project Structure")
    st.code(
        """
app_reccomandation system/
├── app.py
├── requirements.txt
├── data/
│   └── content_df.csv
├── models/
│   ├── count_vectorizer.pkl
│   └── count_matrix.pkl
└── outputs/
    ├── evaluation_results.csv
    ├── advanced_evaluation_results.csv
    ├── model_comparison_results.csv
    ├── model_quality_summary.csv
    ├── popular_movies_baseline.csv
    ├── weighted_movies_baseline.csv
    └── hybrid_ranked_movies.csv
        """
    )
