from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / 'artifacts'

st.set_page_config(page_title='Movie Recommender', page_icon='🎬', layout='wide')


@st.cache_data
def load_content_df() -> pd.DataFrame:
    df = pd.read_csv(ARTIFACTS_DIR / 'content_df.csv')
    for col in ['genres_list', 'keywords_list', 'cast_list', 'director']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


@st.cache_resource
def load_similarity():
    cosine_sim = joblib.load(ARTIFACTS_DIR / 'cosine_sim.pkl')
    title_index = joblib.load(ARTIFACTS_DIR / 'title_index.pkl')
    return cosine_sim, title_index


@st.cache_data
def load_rankings(name: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / name)


def poster_url(path: str | float) -> str | None:
    if isinstance(path, str) and path.strip():
        return f'https://image.tmdb.org/t/p/w500{path}'
    return None


def recommend_movies(title: str, top_n: int = 10) -> pd.DataFrame:
    df = load_content_df()
    cosine_sim, title_index = load_similarity()
    if title not in title_index:
        raise KeyError(f'{title} not found in dataset')

    idx = title_index[title]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    movie_indices = [i[0] for i in scores]
    result = df.loc[movie_indices, [
        'original_title', 'genres_list', 'vote_average', 'popularity', 'overview', 'poster_path'
    ]].copy()
    result['similarity_score'] = [round(float(s[1]), 4) for s in scores]
    return result.reset_index(drop=True)


st.title('🎬 Enhanced Content-Based Movie Recommender')
st.write(
    'This app recommends movies using genres, keywords, cast, director, and overview metadata '
    'from the TMDB 5000 dataset.'
)

tab1, tab2, tab3 = st.tabs(['Recommendations', 'Top Ranked', 'About'])

with tab1:
    df = load_content_df()
    movie_title = st.selectbox('Choose a movie', sorted(df['original_title'].dropna().unique().tolist()))
    top_n = st.slider('Number of recommendations', 5, 15, 10)

    if st.button('Recommend', type='primary'):
        try:
            recs = recommend_movies(movie_title, top_n=top_n)
            st.subheader(f'Movies similar to {movie_title}')
            for _, row in recs.iterrows():
                cols = st.columns([1, 3])
                with cols[0]:
                    url = poster_url(row.get('poster_path'))
                    if url:
                        st.image(url, use_container_width=True)
                with cols[1]:
                    st.markdown(f"### {row['original_title']}")
                    st.write(f"**Genres:** {row['genres_list']}")
                    st.write(f"**Rating:** {row['vote_average']}")
                    st.write(f"**Popularity:** {row['popularity']}")
                    st.write(f"**Similarity Score:** {row['similarity_score']}")
                    st.write(str(row['overview'])[:350] + ('...' if len(str(row['overview'])) > 350 else ''))
                    st.divider()
        except KeyError as exc:
            st.error(str(exc))

with tab2:
    ranking_choice = st.radio(
        'Choose a ranking view',
        ['Weighted Ranking', 'Popularity Ranking', 'Hybrid Ranking'],
        horizontal=True,
    )
    file_map = {
        'Weighted Ranking': 'weighted_ranking.csv',
        'Popularity Ranking': 'popularity_ranking.csv',
        'Hybrid Ranking': 'hybrid_ranking.csv',
    }
    ranked = load_rankings(file_map[ranking_choice]).head(20)
    st.dataframe(ranked, use_container_width=True)

with tab3:
    st.markdown(
        '''
        ### Project summary
        This deployed application is based on an enhanced content-based filtering approach.

        **Core features**
        - merges movie and credit datasets
        - extracts genres, keywords, cast, and director metadata
        - creates a combined text representation for each movie
        - uses CountVectorizer and cosine similarity for recommendations
        - includes ranked movie views using weighted average, popularity, and a hybrid score

        **Suggested deployment**
        - Streamlit Community Cloud
        - Render
        - Hugging Face Spaces
        '''
    )
