import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Recomendador IA", page_icon="🎬", layout="wide")
DATA_DIR, MAX_ROWS = "data", 15000
AF = ["danceability", "energy", "valence", "tempo", "acousticness", "speechiness", "liveness", "loudness"]

# ------------------------- DATOS DE EJEMPLO -------------------------
MOVIES = """title|genres|overview
Matrix|Action SciFi|Un hacker descubre que el mundo es una simulación creada por máquinas y se une a la rebelión.
Inception|Action SciFi Thriller|Un ladrón entra en los sueños de otros para robar secretos y plantar una idea.
Interstellar|Adventure Drama SciFi|Astronautas viajan por un agujero de gusano para encontrar un nuevo hogar para la humanidad.
Blade Runner 2049|SciFi Thriller|Un cazador de androides descubre un secreto que puede destruir el futuro de la humanidad.
Avatar|Action Adventure SciFi|Un exsoldado se une a los habitantes de un planeta lejano y debe elegir un bando.
Toy Story|Animation Family Comedy|Los juguetes de un niño cobran vida cuando nadie los ve y viven aventuras para volver a casa.
Shrek|Animation Family Comedy|Un ogro y un burro parlante rescatan a una princesa de un dragón.
Coco|Animation Family Music|Un niño viaja al mundo de los muertos para descubrir la historia de su familia y su música.
Soul|Animation Family Music|Un profesor de música descubre el sentido de la vida en un viaje entre mundos.
Joker|Crime Drama Thriller|Un comediante fracasado se transforma en un criminal en una ciudad violenta.
El Padrino|Crime Drama|La saga de una familia mafiosa y el ascenso de su heredero al poder.
Pulp Fiction|Crime Thriller Comedy|Historias cruzadas de criminales en Los Ángeles con humor negro y violencia.
Gladiator|Action Drama History|Un general romano traicionado busca venganza convertido en gladiador.
Titanic|Drama Romance|Un romance imposible entre dos jóvenes de distinta clase a bordo de un barco que se hunde.
La La Land|Romance Drama Music|Una actriz y un pianista persiguen sus sueños y se enamoran en Los Ángeles.
Coherence|Thriller SciFi Mystery|Una cena entre amigos se vuelve inquietante cuando pasa un cometa y la realidad se duplica.
Parásitos|Drama Thriller Comedy|Una familia pobre se infiltra en la vida de una familia rica con consecuencias inesperadas.
Mad Max Fury Road|Action Adventure SciFi|Una guerrera y un prisionero huyen por el desierto de un tirano en un mundo postapocalíptico."""
SERIES = """title|genres|overview
Breaking Bad|Crime Drama Thriller|Un profesor de química con cáncer fabrica drogas para asegurar el futuro de su familia.
Better Call Saul|Crime Drama|Un abogado de poca monta se transforma en el astuto defensor de criminales.
Stranger Things|Drama Fantasy SciFi Mystery|Un grupo de niños enfrenta fuerzas sobrenaturales en un pueblo de los años ochenta.
Dark|Drama Mystery SciFi|Desapariciones de niños revelan viajes en el tiempo y secretos entre cuatro familias.
Black Mirror|Drama SciFi Thriller|Historias independientes sobre los peligros de la tecnología en la sociedad.
Friends|Comedy Romance|Seis amigos viven sus amores y trabajos en Nueva York.
The Office|Comedy|La vida diaria de los empleados de una oficina de papel contada como falso documental.
The Crown|Drama History|La vida de la reina Isabel II y los eventos que marcaron su reinado.
Narcos|Crime Drama History|El ascenso y la caída de los grandes narcotraficantes colombianos.
La Casa de Papel|Crime Action Thriller|Un grupo de ladrones planea el atraco perfecto a la fábrica de moneda.
Game of Thrones|Drama Fantasy Action|Casas nobles luchan por el control del Trono de Hierro mientras una amenaza crece en el norte.
Rick and Morty|Animation Comedy SciFi|Un científico loco arrastra a su nieto a aventuras por el multiverso."""
SONGS = """track_name|artists|danceability|energy|valence|tempo|acousticness|speechiness|liveness|loudness
Despacito|Luis Fonsi|.66|.79|.84|178|.20|.12|.07|-4.8
Shape of You|Ed Sheeran|.83|.65|.93|96|.58|.08|.09|-3.2
Blinding Lights|The Weeknd|.51|.80|.33|171|.00|.06|.09|-5.9
Someone Like You|Adele|.56|.32|.29|135|.89|.03|.10|-8.0
Bohemian Rhapsody|Queen|.39|.40|.23|72|.29|.05|.24|-9.9
Smells Like Teen Spirit|Nirvana|.50|.91|.72|117|.00|.05|.11|-4.6
Billie Jean|Michael Jackson|.92|.65|.89|117|.02|.04|.04|-3.0
Bad Guy|Billie Eilish|.70|.43|.56|135|.33|.38|.10|-10.9
Lose Yourself|Eminem|.69|.74|.06|171|.00|.26|.36|-4.6
Con Calma|Daddy Yankee|.74|.86|.65|94|.11|.06|.07|-2.7
Dákiti|Bad Bunny|.73|.57|.14|110|.40|.06|.11|-9.1
Levitating|Dua Lipa|.70|.83|.92|103|.05|.06|.07|-3.9
Stairway to Heaven|Led Zeppelin|.34|.34|.20|82|.58|.04|.12|-12.0
Imagine|John Lennon|.55|.26|.17|76|.91|.03|.07|-12.3
Rolling in the Deep|Adele|.73|.77|.52|105|.14|.03|.05|-5.3
Thunderstruck|AC/DC|.50|.89|.29|134|.00|.04|.13|-5.1
Uptown Funk|Bruno Mars|.86|.61|.93|115|.01|.08|.03|-7.0
Perfect|Ed Sheeran|.60|.45|.52|96|.16|.03|.11|-6.3
Hips Don't Lie|Shakira|.78|.82|.76|100|.28|.07|.28|-4.3
Livin' on a Prayer|Bon Jovi|.53|.89|.81|123|.07|.04|.28|-4.5
Fix You|Coldplay|.21|.43|.12|138|.17|.03|.11|-9.8
Mi Gente|J Balvin|.54|.77|.34|105|.10|.06|.12|-4.9"""


def demo(text):
    return pd.read_csv(io.StringIO(text), sep="|")


@st.cache_data
def load_titles(kind):
    path = os.path.join(DATA_DIR, f"{kind}.csv")
    is_demo = not os.path.exists(path)
    df = demo(MOVIES if kind == "peliculas" else SERIES) if is_demo else pd.read_csv(path)
    df = df.dropna(subset=["title"]).drop_duplicates("title").head(MAX_ROWS)
    for c in ("genres", "overview"):
        df[c] = df[c].fillna("").astype(str)
    return df.reset_index(drop=True), is_demo


@st.cache_data
def load_songs():
    path = os.path.join(DATA_DIR, "canciones.csv")
    is_demo = not os.path.exists(path)
    df = demo(SONGS) if is_demo else pd.read_csv(path)
    df = df.dropna(subset=["track_name", "artists"] + AF).drop_duplicates(["track_name", "artists"]).head(MAX_ROWS)
    df = df.reset_index(drop=True)
    df["label"] = df["track_name"] + " — " + df["artists"].astype(str)
    return df, is_demo


@st.cache_resource
def text_models(kind, n_rows, w, _df):
    gv, ov = TfidfVectorizer(), TfidfVectorizer(max_features=5000)
    G, O = gv.fit_transform(_df["genres"]), ov.fit_transform(_df["overview"])
    # filas normalizadas: el producto punto = w*sim_géneros + (1-w)*sim_sinopsis
    M = hstack([G * np.sqrt(w), O * np.sqrt(1 - w)]).tocsr()
    return M, O, np.array(ov.get_feature_names_out())


@st.cache_resource
def music_model(n_rows, k, _df):
    X = StandardScaler().fit_transform(_df[AF])
    return X, KMeans(k, n_init=10, random_state=42).fit_predict(X), PCA(2, random_state=42).fit_transform(X), NearestNeighbors().fit(X)


def card(rank, title, pct, tags, why):
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{rank}. {title}**  \n`{tags}`")
        c2.metric("Coincidencia", f"{pct:.0f}%")
        st.progress(min(max(pct / 100, 0.0), 1.0))
        if why:
            st.caption("💡 " + why)


# ------------------------- PÁGINAS -------------------------
def home():
    st.title("🎬🎵 Recomendador inteligente")
    st.markdown("Dime qué te gustó y te sugiero qué ver o escuchar, **explicando por qué**.")
    c1, c2, c3 = st.columns(3)
    c1.info("**🎬 Películas y 📺 Series**\n\nCompara géneros y sinopsis con **TF-IDF + similitud coseno**: mide qué tan parecidos son los textos.")
    c2.info("**🎵 Música**\n\nCompara el *sonido* (energía, ánimo, bailabilidad, tempo…) con **k-NN**, y agrupa canciones con **K-Means**.")
    c3.info("**🧠 Perfil de gustos**\n\nPuedes elegir **varios favoritos**: la app promedia tus gustos y busca lo más cercano a ese perfil.")
    st.caption("Elige una sección en el menú de la izquierda 👈")


def titles_page(kind, noun):
    df, is_demo = load_titles(kind)
    st.header(f"{'🎬' if kind == 'peliculas' else '📺'} Recomendador de {kind}")
    if is_demo:
        st.info(f"Estás viendo un catálogo de ejemplo. Sube `data/{kind}.csv` al repositorio para usar datos reales.")
    with st.sidebar:
        st.subheader("⚙️ Ajustes")
        n = st.slider("Cantidad de recomendaciones", 3, 15, 5)
        w = st.slider("Peso de los géneros (vs. sinopsis)", 0.1, 0.9, 0.4, 0.1,
                      help="Más peso a géneros = títulos del mismo tipo. Más peso a sinopsis = historias parecidas.")
    favs = st.multiselect(f"Elige {noun} que te gustaron (puedes elegir varias):", df["title"].tolist(),
                          default=df["title"].tolist()[:1], max_selections=5)
    if not favs:
        st.warning("Elige al menos un título para empezar.")
        return
    M, O, terms = text_models(kind, len(df), w, df)
    idx = df.index[df["title"].isin(favs)].tolist()
    profile = np.asarray(M[idx].mean(axis=0))
    score = linear_kernel(profile, M).ravel()
    score = score / max(score.max(), 1e-9)
    score[idx] = -1
    top = np.argsort(score)[::-1][:n]
    fav_genres = set(" ".join(df.loc[idx, "genres"]).split())
    prof_o = np.asarray(O[idx].mean(axis=0)).ravel()
    tab1, tab2 = st.tabs(["✨ Recomendaciones", "📊 Comparación"])
    with tab1:
        for r, i in enumerate(top, 1):
            shared = sorted(fav_genres & set(df.loc[i, "genres"].split()))
            common = O[i].toarray().ravel() * prof_o
            words = [terms[j] for j in np.argsort(common)[::-1][:3] if common[j] > 0]
            why = []
            if shared:
                why.append("Comparte géneros: " + ", ".join(shared))
            if words:
                why.append("palabras en común: " + ", ".join(words))
            card(r, df.loc[i, "title"], score[i] * 100, df.loc[i, "genres"], " · ".join(why))
    with tab2:
        res = pd.DataFrame({"título": df.loc[top, "title"], "coincidencia (%)": (score[top] * 100).round(1)})
        st.plotly_chart(px.bar(res.sort_values("coincidencia (%)"), x="coincidencia (%)", y="título",
                               orientation="h", title="¿Qué tan parecido es cada título a tus gustos?"), width="stretch")


def music_page():
    df, is_demo = load_songs()
    st.header("🎵 Recomendador de música")
    if is_demo:
        st.info("Catálogo de ejemplo (valores de audio aproximados, solo demostrativos). Sube `data/canciones.csv` para usar datos reales.")
    with st.sidebar:
        st.subheader("⚙️ Ajustes")
        n = st.slider("Cantidad de recomendaciones", 3, 15, 5)
        k = st.slider("Grupos de canciones (K-Means)", 2, 8, 4)
    X, labels, coords, nn = music_model(len(df), k, df)
    tab1, tab2, tab3 = st.tabs(["🎧 Por canciones que me gustan", "😊 Por estado de ánimo", "🗺️ Mapa musical"])
    with tab1:
        favs = st.multiselect("Elige canciones que te gustan:", df["label"].tolist(), default=df["label"].tolist()[:1], max_selections=5)
        if favs:
            idx = df.index[df["label"].isin(favs)].tolist()
            _, nb = nn.kneighbors(X[idx].mean(axis=0).reshape(1, -1), n_neighbors=n + len(idx))
            rec = [i for i in nb[0] if i not in idx][:n]
            d = np.linalg.norm(X[rec] - X[idx].mean(axis=0), axis=1)
            pct = 100 * (1 - d / (d.max() + 1e-9) * 0.7)
            for r, (i, p) in enumerate(zip(rec, pct), 1):
                card(r, df.loc[i, "label"], p, f"energía {df.loc[i, 'energy']:.2f} · ánimo {df.loc[i, 'valence']:.2f}", f"Pertenece al grupo {labels[i] + 1}")
            cols = ["danceability", "energy", "valence", "acousticness", "liveness"]
            fig = go.Figure()
            for name, rows in [("Tus gustos", idx), ("Recomendadas", rec)]:
                fig.add_trace(go.Scatterpolar(r=df.loc[rows, cols].mean().tolist() + [df.loc[rows, cols[0]].mean()],
                                              theta=cols + [cols[0]], fill="toself", name=name))
            fig.update_layout(title="Perfil de sonido: tus gustos vs. recomendadas")
            st.plotly_chart(fig, width="stretch")
    with tab2:
        st.write("Mueve los controles según cómo te sientes y te busco canciones así:")
        c = st.columns(4)
        e, v = c[0].slider("⚡ Energía", 0.0, 1.0, 0.7), c[1].slider("😊 Alegría", 0.0, 1.0, 0.7)
        dn, ac = c[2].slider("💃 Bailable", 0.0, 1.0, 0.7), c[3].slider("🎸 Acústica", 0.0, 1.0, 0.2)
        mood = df[["energy", "valence", "danceability", "acousticness"]].to_numpy()
        dist = np.linalg.norm(mood - np.array([e, v, dn, ac]), axis=1)
        for r, i in enumerate(np.argsort(dist)[:n], 1):
            card(r, df.loc[i, "label"], 100 * (1 - dist[i] / 2), f"tempo {df.loc[i, 'tempo']:.0f} bpm", "")
    with tab3:
        p = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "grupo": (labels + 1).astype(str), "canción": df["label"]})
        st.plotly_chart(px.scatter(p, x="x", y="y", color="grupo", hover_name="canción", opacity=0.7,
                                   title="Cada punto es una canción; las cercanas suenan parecido"), width="stretch")


st.sidebar.title("📋 Menú")
page = st.sidebar.radio("Sección", ["🏠 Inicio", "🎬 Películas", "📺 Series", "🎵 Música"], label_visibility="collapsed")
if page == "🏠 Inicio":
    home()
elif page == "🎬 Películas":
    titles_page("peliculas", "películas")
elif page == "📺 Series":
    titles_page("series", "series")
else:
    music_page()
