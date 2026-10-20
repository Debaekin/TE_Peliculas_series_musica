import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Recomendador", page_icon="🎬", layout="wide")

DATA_DIR = "data"
MAX_ROWS = 15000  # límite para que la app sea rápida en Streamlit Cloud
AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "speechiness", "liveness", "loudness",
]

# ---------------------------------------------------------------------------
# DATOS DE DEMOSTRACIÓN (se usan solo si no existen los CSV reales en /data)
# ---------------------------------------------------------------------------
DEMO_MOVIES = pd.DataFrame({
    "title": ["Matrix", "Inception", "Interstellar", "Toy Story", "Shrek",
              "Coco", "Joker", "Gladiator", "Titanic", "Avatar"],
    "genres": ["Action Science Fiction", "Action Science Fiction Thriller",
               "Adventure Drama Science Fiction", "Animation Family Comedy",
               "Animation Family Comedy", "Animation Family Music",
               "Crime Drama Thriller", "Action Drama History",
               "Drama Romance", "Action Adventure Science Fiction"],
    "overview": [
        "Un hacker descubre que la realidad es una simulación controlada por máquinas.",
        "Un ladrón entra en los sueños de otros para robar secretos y plantar ideas.",
        "Astronautas viajan a través de un agujero de gusano para salvar a la humanidad.",
        "Los juguetes de un niño cobran vida cuando nadie los ve.",
        "Un ogro y un burro parlante rescatan a una princesa de un dragón.",
        "Un niño viaja al mundo de los muertos para descubrir la historia de su familia.",
        "Un comediante fracasado se transforma en un criminal en una ciudad violenta.",
        "Un general romano traicionado busca venganza como gladiador.",
        "Un romance imposible a bordo de un barco que se hunde.",
        "Un exsoldado se une a una misión en un planeta lejano habitado por seres azules.",
    ],
})
DEMO_SERIES = pd.DataFrame({
    "title": ["Breaking Bad", "Stranger Things", "The Crown", "Dark",
              "Friends", "Black Mirror", "Narcos", "La Casa de Papel"],
    "genres": ["Crime Drama Thriller", "Drama Fantasy Science Fiction Mystery",
               "Drama History", "Drama Mystery Science Fiction",
               "Comedy Romance", "Drama Science Fiction Thriller",
               "Crime Drama History", "Crime Action Thriller"],
    "overview": [
        "Un profesor de química con cáncer fabrica drogas para asegurar el futuro de su familia.",
        "Un grupo de niños enfrenta fuerzas sobrenaturales en un pueblo de los años ochenta.",
        "La vida de la reina Isabel II y los eventos que marcaron su reinado.",
        "Desapariciones de niños revelan viajes en el tiempo y secretos entre familias.",
        "Seis amigos viven sus amores y trabajos en Nueva York.",
        "Historias independientes sobre los peligros de la tecnología en la sociedad.",
        "El ascenso y caída de los grandes narcotraficantes colombianos.",
        "Un grupo de ladrones planea el atraco perfecto a la fábrica de moneda.",
    ],
})


def make_demo_songs(n=300):
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.random((n, len(AUDIO_FEATURES))), columns=AUDIO_FEATURES)
    df["tempo"] = df["tempo"] * 120 + 70
    df["loudness"] = df["loudness"] * -30
    df["track_name"] = [f"Canción demo {i + 1}" for i in range(n)]
    df["artists"] = [f"Artista {i % 25 + 1}" for i in range(n)]
    return df


# ---------------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------------
@st.cache_data
def load_titles(kind):
    """kind = 'peliculas' o 'series'. CSV esperado: title, genres, overview."""
    path = os.path.join(DATA_DIR, f"{kind}.csv")
    is_demo = not os.path.exists(path)
    if is_demo:
        df = DEMO_MOVIES.copy() if kind == "peliculas" else DEMO_SERIES.copy()
    else:
        df = pd.read_csv(path)
    df = df.dropna(subset=["title"]).drop_duplicates("title")
    df["genres"] = df["genres"].fillna("").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)
    return df.head(MAX_ROWS).reset_index(drop=True), is_demo


@st.cache_data
def load_songs():
    """CSV esperado: track_name, artists + las columnas de AUDIO_FEATURES."""
    path = os.path.join(DATA_DIR, "canciones.csv")
    is_demo = not os.path.exists(path)
    df = make_demo_songs() if is_demo else pd.read_csv(path)
    df = df.dropna(subset=["track_name", "artists"] + AUDIO_FEATURES)
    df = df.drop_duplicates(["track_name", "artists"]).head(MAX_ROWS).reset_index(drop=True)
    df["label"] = df["track_name"] + " — " + df["artists"].astype(str)
    return df, is_demo


# ---------------------------------------------------------------------------
# MODELOS
# ---------------------------------------------------------------------------
@st.cache_resource
def build_text_models(kind, n_rows, _df):
    """TF-IDF por separado para géneros y sinopsis (el peso se ajusta en la UI)."""
    genre_vec = TfidfVectorizer()
    overview_vec = TfidfVectorizer(max_features=5000)
    return (
        genre_vec.fit_transform(_df["genres"]),
        overview_vec.fit_transform(_df["overview"]),
    )


def recommend_titles(df, genre_m, overview_m, title, n, w_genre):
    idx = df.index[df["title"] == title][0]
    sim_g = linear_kernel(genre_m[idx], genre_m).ravel()
    sim_o = linear_kernel(overview_m[idx], overview_m).ravel()
    score = w_genre * sim_g + (1 - w_genre) * sim_o
    score[idx] = -1  # excluye el mismo título
    top = np.argsort(score)[::-1][:n]
    out = df.loc[top, ["title", "genres"]].copy()
    out["similitud"] = score[top].round(3)
    return out.reset_index(drop=True)


@st.cache_resource
def build_music_model(n_rows, k, _df):
    X = StandardScaler().fit_transform(_df[AUDIO_FEATURES])
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    nn = NearestNeighbors(metric="euclidean").fit(X)
    return X, labels, coords, nn


# ---------------------------------------------------------------------------
# INTERFAZ
# ---------------------------------------------------------------------------
st.title("🎬🎵 Recomendador de películas, series y música")
st.sidebar.title("📋 Navegación")
section = st.sidebar.selectbox("Qué quieres explorar:", ["🎬 Películas", "📺 Series", "🎵 Música"])


def titles_page(kind, label):
    df, is_demo = load_titles(kind)
    if is_demo:
        st.info(f"Usando datos de demostración. Sube `data/{kind}.csv` al repositorio para usar datos reales.")
    genre_m, overview_m = build_text_models(kind, len(df), df)

    col1, col2 = st.columns([2, 1])
    with col1:
        title = st.selectbox(f"Elige {label} que te gustó:", df["title"].tolist())
    with col2:
        n = st.slider("Número de recomendaciones", 3, 20, 5)
    w_genre = st.slider("Peso de los géneros vs. la sinopsis", 0.0, 1.0, 0.4, 0.1)

    recs = recommend_titles(df, genre_m, overview_m, title, n, w_genre)
    st.subheader(f"Si te gustó «{title}», prueba con:")
    st.dataframe(recs, width='stretch', hide_index=True)

    fig = px.bar(recs.sort_values("similitud"), x="similitud", y="title",
                 orientation="h", title="Similitud de las recomendaciones")
    st.plotly_chart(fig, width='stretch')


def music_page():
    df, is_demo = load_songs()
    if is_demo:
        st.info("Usando datos de demostración. Sube `data/canciones.csv` al repositorio para usar datos reales.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        label = st.selectbox("Elige una canción que te gustó:", df["label"].tolist())
    with col2:
        n = st.slider("Recomendaciones", 3, 20, 5)
    with col3:
        k = st.slider("Número de clusters (K-Means)", 2, 10, 5)

    X, labels, coords, nn = build_music_model(len(df), k, df)
    idx = df.index[df["label"] == label][0]
    _, neighbors = nn.kneighbors(X[idx].reshape(1, -1), n_neighbors=n + 1)
    neighbors = [i for i in neighbors[0] if i != idx][:n]

    st.subheader(f"Si te gustó «{label}», prueba con:")
    st.dataframe(df.loc[neighbors, ["track_name", "artists"] + AUDIO_FEATURES],
                 width='stretch', hide_index=True)

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1],
        "cluster": labels.astype(str), "canción": df["label"],
    })
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", hover_name="canción",
                     opacity=0.6, title="Mapa de canciones por similitud de audio (PCA)")
    sel = plot_df.iloc[[idx]]
    fig.add_scatter(x=sel["PC1"], y=sel["PC2"], mode="markers", name="Tu canción",
                    marker=dict(size=16, color="black", symbol="star"))
    st.plotly_chart(fig, width='stretch')


if section == "🎬 Películas":
    titles_page("peliculas", "una película")
elif section == "📺 Series":
    titles_page("series", "una serie")
else:
    music_page()
